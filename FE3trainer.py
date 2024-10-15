import torch
import argparse
import logging
import os, sys
import time, statistics
from pathlib import Path
import numpy as np
from sklearn import metrics
from utils import trainUtils
from modules import FE3mask

parser = argparse.ArgumentParser(description="multiemb trainer")
parser.add_argument("--dataset", type=str, help="specify dataset", default="ali-ccp")
parser.add_argument("--model", type=str, help="specify model", default="dnn")


# training hyperparameters
parser.add_argument("--lr", type=float, help="learning rate", default=3e-4)
parser.add_argument("--l2", type=float, help="L2 regularization", default=3e-6)
parser.add_argument("--bsize", type=int, help="batchsize", default=4096)
parser.add_argument("--optim", type=str, default="Adam", help="optimizer type")
parser.add_argument("--max_epoch", type=int, default=20, help="maxmium epochs")
parser.add_argument("--save_dir", type=Path, default="save/", help="model save directory")

# neural network hyperparameters
parser.add_argument("--dim", type=int, help="embedding dimension", default=16)
parser.add_argument("--mlp_dims", type=int, nargs='+', default=[1024, 512, 256], help="mlp layer size")
parser.add_argument("--mlp_dropout", type=float, default=0.0, help="mlp dropout rate (default:0.0)")
parser.add_argument("--mlp_bn", action="store_true", default=False, help="mlp batch normalization")
parser.add_argument("--cross", type=int, help="cross layer", default=3)

# device information
parser.add_argument("--cuda", type=int, choices=range(-1, 8), default=0, help="device info")

# mask information
parser.add_argument("--final_temp", type=float, default=500, help="final temperature")
parser.add_argument("--search_epoch", type=int, default=1, help="search epochs")
parser.add_argument("--rewind", type=int, default=1, help="rewind model")
parser.add_argument("--scaling", type=float, default=1, help="soft mask scaling")
parser.add_argument("--lambda1", type=float, default=3, help="share lambda")
args = parser.parse_args()

my_seed = 2022
torch.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)
np.random.seed(my_seed)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
os.environ['NUMEXPR_MAX_THREADS'] = '8'


class Trainer(object):
    def __init__(self, opt):
        self.opt = opt
        self.model_opt = opt["model_opt"]
        self.lr = opt['lr']
        self.l2 = opt['l2']
        self.bs = opt['bsize']
        self.model_dir = opt["save_dir"]
        self.epochs = opt["search_epoch"]
        self.rewind = opt["rewind"]
        self.lambda1 = opt["lambda1"]
        self.dataloader = trainUtils.getDataLoader(opt["dataset"], opt["data_dir"])
        self.device = trainUtils.getDevice(opt["cuda"])
        self.network = FE3mask.getModel(opt["model"], opt["model_opt"]).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.optim = FE3mask.getOptim(self.network, opt["optimizer"], self.lr, self.l2)
        self.logger = trainUtils.get_log(opt['model'])
        self.model_init_path = 'model3_init'
        self.mask_params_path = 'mask3_params.pth'
        torch.save(self.network.state_dict(), self.model_init_path)

        ########
        params = self.network.domain_hypernet.parameters()
        total_sum = sum(param.sum().item() for param in params)
        print('domain_hypernet params sum 1:', total_sum)
        ########


    def train_on_batch(self, label, data, domain, retrain=False):
        self.network.train()
        self.network.zero_grad()
        data, label, domain = data.to(self.device), label.to(self.device), domain.to(self.device)
        logit1, logit2, logit3 = self.network(data, domain)
        logloss1 = self.criterion(logit1, label)
        logloss2 = self.criterion(logit2, label)
        logloss3 = self.criterion(logit3, label)

        if not retrain:
            share_logloss = torch.mean(logloss1 + logloss2 + logloss3) / 3
            domain_logloss = torch.mean(logloss1 * torch.eq(domain, 0).type(torch.long) + logloss2 * torch.eq(domain, 1).type(
                    torch.long) + logloss3 * torch.eq(domain, 2).type(torch.long))
            loss = domain_logloss + self.lambda1 * share_logloss
        else:
            share_logloss = torch.mean(logloss1 + logloss2 + logloss3) / 3
            domain_logloss = torch.mean(logloss1 * torch.eq(domain, 0).type(torch.long) + logloss2 * torch.eq(domain, 1).type(
                    torch.long) + logloss3 * torch.eq(domain, 2).type(torch.long))
            loss = domain_logloss + self.lambda1 * share_logloss
        loss.backward()
        for optim in self.optim:
            optim.step()
        return loss.item()

    def eval_on_batch(self, data, domain):
        self.network.eval()
        with torch.no_grad():
            data, domain = data.to(self.device), domain.to(self.device)
            logit1, logit2, logit3 = self.network(data, domain)
            if domain[0] == 0:
                logit = logit1
            elif domain[0] == 1:
                logit = logit2
            elif domain[0] == 2:
                logit = logit3
            else:
                print("domain error")
            prob = torch.sigmoid(logit).detach().cpu().numpy()
        return prob

    def search(self):
        self.logger.info("ticket:{t}".format(t=self.network.ticket))
        self.logger.info("-----------------Begin Search-----------------")
        self.ds = self.dataloader.get_train_data("train", batch_size=self.bs)
        temp_increase = self.opt["final_temp"] ** (1. / (len(self.ds) - 1))
        for epoch_idx in range(int(self.epochs)):
            train_loss = .0
            step = 0
            for feature, label, domain in self.ds:
                if step > 0:
                    self.network.temp *= temp_increase
                loss = self.train_on_batch(label, feature, domain)
                train_loss += loss
                step += 1
                if step % 1000 == 0:
                    print('fmask1_rate:', sum(self.network.fmask1_rate_list) / len(self.network.fmask1_rate_list),
                          'fmask2_rate:', sum(self.network.fmask2_rate_list) / len(self.network.fmask2_rate_list),
                          'fmask3_rate:', sum(self.network.fmask3_rate_list) / len(self.network.fmask3_rate_list))
                    print('emask1_rate:', sum(self.network.emask1_rate_list) / len(self.network.emask1_rate_list),
                          'emask2_rate:', sum(self.network.emask2_rate_list) / len(self.network.emask2_rate_list),
                          'emask3_rate:', sum(self.network.emask3_rate_list) / len(self.network.emask3_rate_list))
                    self.logger.info("[Epoch {epoch:d} | Step :{setp:d} | Train Loss:{loss:.6f}".
                                     format(epoch=epoch_idx, setp=step, loss=loss))
            train_loss /= step
            self.logger.info("Temp:{temp:.6f}".format(temp=self.network.temp))
            val_auc, val_loss = self.evaluate_val("validation")
            print('fmask1_rate:', sum(self.network.fmask1_rate_list) / len(self.network.fmask1_rate_list),
                  'fmask2_rate:', sum(self.network.fmask2_rate_list) / len(self.network.fmask2_rate_list),
                  'fmask3_rate:', sum(self.network.fmask3_rate_list) / len(self.network.fmask3_rate_list))
            print('emask1_rate:', sum(self.network.emask1_rate_list) / len(self.network.emask1_rate_list),
                  'emask2_rate:', sum(self.network.emask2_rate_list) / len(self.network.emask2_rate_list),
                  'emask3_rate:', sum(self.network.emask3_rate_list) / len(self.network.emask3_rate_list))
            self.network.fmask1_rate_list = []
            self.network.fmask2_rate_list = []
            self.network.fmask3_rate_list = []
            self.network.emask1_rate_list = []
            self.network.emask2_rate_list = []
            self.network.emask3_rate_list = []
            self.logger.info(
                "[Epoch {epoch:d} | Train Loss: {loss:.6f} | Val AUC: {val_auc:.6f}, Val Loss: {val_loss:.6f}]".format(
                    epoch=epoch_idx, loss=train_loss, val_auc=val_auc, val_loss=val_loss))

        test_auc0, test_loss0 = self.evaluate_test("test", "0")
        test_auc1, test_loss1 = self.evaluate_test("test", "1")
        test_auc2, test_loss2 = self.evaluate_test("test", "2")
        self.logger.info(
            "Test AUC0: {test_auc:.6f}, Test Loss0: {test_loss:.6f}".format(test_auc=test_auc0, test_loss=test_loss0))
        self.logger.info(
            "Test AUC1: {test_auc:.6f}, Test Loss1: {test_loss:.6f}".format(test_auc=test_auc1, test_loss=test_loss1))
        self.logger.info(
            "Test AUC2: {test_auc:.6f}, Test Loss2: {test_loss:.6f}".format(test_auc=test_auc2, test_loss=test_loss2))

        ########
        params = self.network.domain_hypernet.parameters()
        total_sum = sum(param.sum().item() for param in params)
        print('domain_hypernet params sum 2:', total_sum)
        ########

        self.save_mask_params(self.network, self.rewind)

    def evaluate_val(self, on: str):
        preds, trues = [], []
        for feature, label, domain in self.dataloader.get_data(on + "0", batch_size=self.bs * 10):
            pred = self.eval_on_batch(feature, domain)
            label = label.detach().cpu().numpy()
            preds.append(pred)
            trues.append(label)
        for feature, label, domain in self.dataloader.get_data(on + "1", batch_size=self.bs * 10):
            pred = self.eval_on_batch(feature, domain)
            label = label.detach().cpu().numpy()
            preds.append(pred)
            trues.append(label)
        for feature, label, domain in self.dataloader.get_data(on + "2", batch_size=self.bs * 10):
            pred = self.eval_on_batch(feature, domain)
            label = label.detach().cpu().numpy()
            preds.append(pred)
            trues.append(label)
        y_pred = np.concatenate(preds).astype("float64")
        y_true = np.concatenate(trues).astype("float64")
        auc = metrics.roc_auc_score(y_true, y_pred)
        loss = metrics.log_loss(y_true, y_pred)
        return auc, loss

    def evaluate_test(self, on: str, dom: str):
        preds, trues = [], []
        for feature, label, domain in self.dataloader.get_data(on + dom, batch_size=self.bs * 10):
            pred = self.eval_on_batch(feature, domain)
            label = label.detach().cpu().numpy()
            preds.append(pred)
            trues.append(label)
        y_pred = np.concatenate(preds).astype("float64")
        y_true = np.concatenate(trues).astype("float64")
        auc = metrics.roc_auc_score(y_true, y_pred)
        loss = metrics.log_loss(y_true, y_pred)
        return auc, loss

    def train(self, epochs):
        self.network = FE3mask.getModel(self.opt["model"], self.opt["model_opt"]).to(self.device)
        self.network.load_state_dict(torch.load(self.model_init_path))

        ########
        params = self.network.domain_hypernet.parameters()
        total_sum = sum(param.sum().item() for param in params)
        print('domain_hypernet params sum 3:', total_sum)
        ########

        self.load_mask_params(self.network, self.rewind)
        self.network.ticket = True

        ########
        params = self.network.domain_hypernet.parameters()
        total_sum = sum(param.sum().item() for param in params)
        print('domain_hypernet params sum 4:', total_sum)
        ########

        cur_auc = 0.0
        early_stop = False
        self.optim = FE3mask.getOptim(self.network, "adam", self.lr, self.l2)[:1]

        self.logger.info("-----------------Begin Train-----------------")
        self.logger.info("Ticket:{t}".format(t=self.network.ticket))

        for epoch_idx in range(int(epochs)):
            train_loss = .0
            step = 0
            for feature, label, domain in self.ds:
                loss = self.train_on_batch(label, feature, domain, retrain=True)
                train_loss += loss
                step += 1
                if step % 1000 == 0:
                    print('fmask1_rate:', sum(self.network.fmask1_rate_list) / len(self.network.fmask1_rate_list),
                          'fmask2_rate:', sum(self.network.fmask2_rate_list) / len(self.network.fmask2_rate_list),
                          'fmask3_rate:', sum(self.network.fmask3_rate_list) / len(self.network.fmask3_rate_list))
                    print('emask1_rate:', sum(self.network.emask1_rate_list) / len(self.network.emask1_rate_list),
                          'emask2_rate:', sum(self.network.emask2_rate_list) / len(self.network.emask2_rate_list),
                          'emask3_rate:', sum(self.network.emask3_rate_list) / len(self.network.emask3_rate_list))
                    print('mask_rate:', sum(self.network.mask_rate_list) / len(self.network.mask_rate_list))
                    self.logger.info("[Epoch {epoch:d} | Step :{setp:d} | Train Loss:{loss:.6f}".
                                     format(epoch=epoch_idx, setp=step, loss=loss))
            train_loss /= step
            val_auc, val_loss = self.evaluate_val("validation")
            print('fmask1_rate:', sum(self.network.fmask1_rate_list) / len(self.network.fmask1_rate_list),
                  'fmask2_rate:', sum(self.network.fmask2_rate_list) / len(self.network.fmask2_rate_list),
                  'fmask3_rate:', sum(self.network.fmask3_rate_list) / len(self.network.fmask3_rate_list))
            print('emask1_rate:', sum(self.network.emask1_rate_list) / len(self.network.emask1_rate_list),
                  'emask2_rate:', sum(self.network.emask2_rate_list) / len(self.network.emask2_rate_list),
                  'emask3_rate:', sum(self.network.emask3_rate_list) / len(self.network.emask3_rate_list))
            print('mask_rate:', sum(self.network.mask_rate_list) / len(self.network.mask_rate_list))
            self.network.fmask1_rate_list = []
            self.network.fmask2_rate_list = []
            self.network.fmask3_rate_list = []
            self.network.emask1_rate_list = []
            self.network.emask2_rate_list = []
            self.network.emask3_rate_list = []
            self.network.mask_rate_list = []
            self.logger.info(
                "[Epoch {epoch:d} | Train Loss: {loss:.6f} | Val AUC: {val_auc:.6f}, Val Loss: {val_loss:.6f}]".format(
                    epoch=epoch_idx, loss=train_loss, val_auc=val_auc, val_loss=val_loss))

            if val_auc > cur_auc:
                cur_auc = val_auc
                torch.save(self.network.state_dict(), self.model_dir)
            else:
                self.network.load_state_dict(torch.load(self.model_dir))
                self.network.to(self.device)
                early_stop = True
                test_auc0, test_loss0 = self.evaluate_test("test", "0")
                test_auc1, test_loss1 = self.evaluate_test("test", "1")
                test_auc2, test_loss2 = self.evaluate_test("test", "2")
                self.logger.info(
                    "Early stop at epoch {epoch:d} | Test AUC0: {test_auc:.6f}, Test Loss0:{test_loss:.6f}".format(
                        epoch=epoch_idx, test_auc=test_auc0, test_loss=test_loss0))
                self.logger.info(
                    "Early stop at epoch {epoch:d} | Test AUC1: {test_auc:.6f}, Test Loss1:{test_loss:.6f}".format(
                        epoch=epoch_idx, test_auc=test_auc1, test_loss=test_loss1))
                self.logger.info(
                    "Early stop at epoch {epoch:d} | Test AUC2: {test_auc:.6f}, Test Loss2:{test_loss:.6f}".format(
                        epoch=epoch_idx, test_auc=test_auc2, test_loss=test_loss2))
                break
        if not early_stop:
            test_auc0, test_loss0 = self.evaluate_test("test", "0")
            test_auc1, test_loss1 = self.evaluate_test("test", "1")
            test_auc2, test_loss2 = self.evaluate_test("test", "2")
            self.logger.info(
                "Final Test AUC0: {test_auc:.6f}, Test Loss0: {test_loss:.6f}".format(test_auc=test_auc0,
                                                                                      test_loss=test_loss0))
            self.logger.info(
                "Final Test AUC1: {test_auc:.6f}, Test Loss1: {test_loss:.6f}".format(test_auc=test_auc1,
                                                                                      test_loss=test_loss1))
            self.logger.info(
                "Final Test AUC2: {test_auc:.6f}, Test Loss2: {test_loss:.6f}".format(test_auc=test_auc2,
                                                                                      test_loss=test_loss2))

    def save_mask_params(self, model, rewind):
        if rewind:
            torch.save({
                'embedding_state_dict': model.embedding.state_dict(),
                'domain_hypernet_state_dict': model.domain_hypernet.state_dict(),
                'domain1_fmask_state_dict': model.domain1_fmask.state_dict(),
                'domain2_fmask_state_dict': model.domain2_fmask.state_dict(),
                'domain3_fmask_state_dict': model.domain3_fmask.state_dict(),
                'domain1_emask_state_dict': model.domain1_emask.state_dict(),
                'domain2_emask_state_dict': model.domain2_emask.state_dict(),
                'domain3_emask_state_dict': model.domain3_emask.state_dict()
            }, self.mask_params_path)
        else:
            torch.save({
                'domain_hypernet_state_dict': model.domain_hypernet.state_dict(),
                'domain1_fmask_state_dict': model.domain1_fmask.state_dict(),
                'domain2_fmask_state_dict': model.domain2_fmask.state_dict(),
                'domain3_fmask_state_dict': model.domain3_fmask.state_dict(),
                'domain1_emask_state_dict': model.domain1_emask.state_dict(),
                'domain2_emask_state_dict': model.domain2_emask.state_dict(),
                'domain3_emask_state_dict': model.domain3_emask.state_dict()
            }, self.mask_params_path)

    def load_mask_params(self, model, rewind):
        checkpoint = torch.load(self.mask_params_path)
        if rewind:
            model.embedding.load_state_dict(checkpoint['embedding_state_dict'])
        model.domain_hypernet.load_state_dict(checkpoint['domain_hypernet_state_dict'])
        model.domain1_fmask.load_state_dict(checkpoint['domain1_fmask_state_dict'])
        model.domain2_fmask.load_state_dict(checkpoint['domain2_fmask_state_dict'])
        model.domain3_fmask.load_state_dict(checkpoint['domain3_fmask_state_dict'])
        model.domain1_emask.load_state_dict(checkpoint['domain1_emask_state_dict'])
        model.domain2_emask.load_state_dict(checkpoint['domain2_emask_state_dict'])
        model.domain3_emask.load_state_dict(checkpoint['domain3_emask_state_dict'])

def main():
    sys.path.extend(["./modules", "./dataloader", "./utils"])
    if args.dataset.lower() == "ali-ccp":
        field_dim = trainUtils.get_stats("./data/ali-ccp/stats")
        data_dir = "./data/ali-ccp/tfrecord"
        field = len(field_dim)
        feature = sum(field_dim)
    else:
        print("dataset error")
    model_opt = {
        "latent_dim": args.dim, "feat_num": feature, "field_num": field, "scaling": args.scaling,
        "mlp_dropout": args.mlp_dropout, "use_bn": args.mlp_bn, "mlp_dims": args.mlp_dims, "cross": args.cross
    }

    opt = {
        "model_opt": model_opt, "dataset": args.dataset, "model": args.model, "lr": args.lr, "l2": args.l2,
        "bsize": args.bsize, "optimizer": args.optim, "data_dir": data_dir, "save_dir": args.save_dir,
        "cuda": args.cuda, "search_epoch": args.search_epoch, "rewind": args.rewind,
        "final_temp": args.final_temp, 'lambda1': args.lambda1
    }
    print(opt)
    trainer = Trainer(opt)
    trainer.search()
    trainer.train(args.max_epoch)


if __name__ == "__main__":
    """
    python FE3trainer.py --dataset 'ali-ccp' --model 'deepfm'   
    """
    main()
