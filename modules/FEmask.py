import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.layers import MultiLayerPerceptron, FactorizationMachine, FeatureEmbedding
import modules.layers as layer

class MaskedNet(torch.nn.Module):
    def __init__(self, opt):
        super(MaskedNet, self).__init__()
        self.latent_dim = opt["latent_dim"]
        self.feature_num = opt["feat_num"]
        self.field_num = opt["field_num"]
        self.scaling = opt["scaling"]
        self.embed_dims = opt["mlp_dims"]
        self.dropout = opt["mlp_dropout"]
        self.use_bn = opt["use_bn"]
        self.ticket = False
        self.temp = 1
        print(self.field_num)
        print(self.feature_num)
        self.embedding = FeatureEmbedding(self.feature_num, self.latent_dim)

        self.fmask1_rate_list = []
        self.fmask2_rate_list = []
        self.emask1_rate_list = []
        self.emask2_rate_list = []
        self.mask_rate_list = []
        self.mask1_rate_list = []
        self.mask2_rate_list = []

    def init_masknet(self):
        self.domain_hypernet = MultiLayerPerceptron(self.dnn_dim, self.embed_dims, output_layer=False, dropout=self.dropout,
                                                    use_bn=self.use_bn)
        self.domain1_fmask = torch.nn.Linear(self.embed_dims[-1], self.field_num)
        self.domain2_fmask = torch.nn.Linear(self.embed_dims[-1], self.field_num)
        self.domain1_emask = torch.nn.Linear(self.embed_dims[-1], self.latent_dim)
        self.domain2_emask = torch.nn.Linear(self.embed_dims[-1], self.latent_dim)

    def compute_mask(self, embed):
        x_dnn = embed.view(-1, self.dnn_dim)
        hyper_outpot = self.domain_hypernet(x_dnn)
        if self.ticket:
            fm1 = self.domain1_fmask(hyper_outpot)
            fm2 = self.domain2_fmask(hyper_outpot)
            em1 = self.domain1_emask(hyper_outpot)
            em2 = self.domain2_emask(hyper_outpot)
            fmask1 = (fm1 > 0).float()
            fmask2 = (fm2 > 0).float()
            emask1 = (em1 > 0).float()
            emask2 = (em2 > 0).float()
            self.fmask1_rate_list.append((torch.sum((fm1 > 0).float()) / (fm1.shape[0] * fm1.shape[1])).item())
            self.fmask2_rate_list.append((torch.sum((fm2 > 0).float()) / (fm2.shape[0] * fm2.shape[1])).item())
            self.emask1_rate_list.append((torch.sum((em1 > 0).float()) / (em1.shape[0] * em1.shape[1])).item())
            self.emask2_rate_list.append((torch.sum((em2 > 0).float()) / (em2.shape[0] * em2.shape[1])).item())
        else:
            fm1 = self.domain1_fmask(hyper_outpot)
            fm2 = self.domain2_fmask(hyper_outpot)
            em1 = self.domain1_emask(hyper_outpot)
            em2 = self.domain2_emask(hyper_outpot)
            fmask1 = torch.sigmoid(self.temp * fm1)
            fmask2 = torch.sigmoid(self.temp * fm2)
            emask1 = torch.sigmoid(self.temp * em1)
            emask2 = torch.sigmoid(self.temp * em2)
            self.fmask1_rate_list.append((torch.sum((fm1 > 0).float()) / (fm1.shape[0] * fm1.shape[1])).item())
            self.fmask2_rate_list.append((torch.sum((fm2 > 0).float()) / (fm2.shape[0] * fm2.shape[1])).item())
            self.emask1_rate_list.append((torch.sum((em1 > 0).float()) / (em1.shape[0] * em1.shape[1])).item())
            self.emask2_rate_list.append((torch.sum((em2 > 0).float()) / (em2.shape[0] * em2.shape[1])).item())

        return fmask1, fmask2, emask1, emask2

    def compute_remaining_weights(self, mask1, mask2, d):
        d = d.unsqueeze(2)
        mask = (1 - d) * mask1 + d * mask2
        self.mask_rate_list.append(torch.sum(mask) / (mask.shape[0] * mask.shape[1] * mask.shape[2]))


    def mask_embedding(self, x, d):
        embed = self.embedding(x)
        fmask1, fmask2, emask1, emask2 = self.compute_mask(embed)
        embed1 = embed * self.scaling * fmask1.unsqueeze(2) * emask1.unsqueeze(1)
        embed2 = embed * self.scaling * fmask2.unsqueeze(2) * emask2.unsqueeze(1)

        if self.ticket:
            mask1 = fmask1.unsqueeze(2) * emask1.unsqueeze(1)
            mask2 = fmask2.unsqueeze(2) * emask2.unsqueeze(1)
            self.compute_remaining_weights(mask1, mask2, d)

        return embed1, embed2

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, field_num)``

        """
        pass

    def reg(self):
        return 0.0


class FM(MaskedNet):
    def __init__(self, opt):
        super(FM, self).__init__(opt)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        pass


class MaskDNN(MaskedNet):
    def __init__(self, opt):
        super(MaskDNN, self).__init__(opt)
        embed_dims = opt["mlp_dims"]
        dropout = opt["mlp_dropout"]
        use_bn = opt["use_bn"]
        self.dnn_dim = self.field_num * self.latent_dim
        self.dnn1 = MultiLayerPerceptron(self.dnn_dim, embed_dims, dropout, use_bn=use_bn)
        self.dnn2 = MultiLayerPerceptron(self.dnn_dim, embed_dims, dropout, use_bn=use_bn)
        self.init_masknet()

    def forward(self, x, d):
        x_embedding1, x_embedding2 = self.mask_embedding(x, d)

        x_dnn1 = x_embedding1.view(-1, self.dnn_dim)
        x_dnn2 = x_embedding2.view(-1, self.dnn_dim)

        output_dnn1 = self.dnn1(x_dnn1)
        output_dnn2 = self.dnn2(x_dnn2)
        logit1 = output_dnn1
        logit2 = output_dnn2
        return logit1, logit2


class MaskDeepFM(FM):
    def __init__(self, opt):
        super(MaskDeepFM, self).__init__(opt)
        embed_dims = opt["mlp_dims"]
        dropout = opt["mlp_dropout"]
        use_bn = opt["use_bn"]
        self.dnn_dim = self.field_num * self.latent_dim
        self.dnn1 = MultiLayerPerceptron(self.dnn_dim, embed_dims, dropout, use_bn=use_bn)
        self.dnn2 = MultiLayerPerceptron(self.dnn_dim, embed_dims, dropout, use_bn=use_bn)
        self.init_masknet()

    def forward(self, x, d):
        x_embedding1, x_embedding2 = self.mask_embedding(x, d)

        output_fm1 = self.fm(x_embedding1)
        output_fm2 = self.fm(x_embedding2)
        x_dnn1 = x_embedding1.view(-1, self.dnn_dim)
        x_dnn2 = x_embedding2.view(-1, self.dnn_dim)
        output_dnn1 = self.dnn1(x_dnn1)
        output_dnn2 = self.dnn2(x_dnn2)
        logit1 = output_dnn1 + output_fm1
        logit2 = output_dnn2 + output_fm2
        return logit1, logit2


class MaskDeepCross(MaskedNet):
    def __init__(self, opt):
        super(MaskDeepCross, self).__init__(opt)
        cross_num = opt["cross"]
        mlp_dims = opt["mlp_dims"]
        use_bn = opt["use_bn"]
        dropout = opt["mlp_dropout"]
        self.dnn_dim = self.field_num * self.latent_dim
        self.cross1 = layer.CrossNetwork(self.dnn_dim, cross_num)
        self.dnn1 = MultiLayerPerceptron(self.dnn_dim, mlp_dims, output_layer=False, dropout=dropout, use_bn=use_bn)
        self.combination1 = torch.nn.Linear(mlp_dims[-1] + self.dnn_dim, 1, bias=False)
        self.cross2 = layer.CrossNetwork(self.dnn_dim, cross_num)
        self.dnn2 = MultiLayerPerceptron(self.dnn_dim, mlp_dims, output_layer=False, dropout=dropout, use_bn=use_bn)
        self.combination2 = torch.nn.Linear(mlp_dims[-1] + self.dnn_dim, 1, bias=False)
        self.init_masknet()

    def forward(self, x, d):
        x_embedding1, x_embedding2 = self.mask_embedding(x, d)

        x_dnn1 = x_embedding1.view(-1, self.dnn_dim)
        x_dnn2 = x_embedding2.view(-1, self.dnn_dim)

        output_cross1 = self.cross1(x_dnn1)
        output_dnn1 = self.dnn1(x_dnn1)
        comb_tensor1 = torch.cat((output_cross1, output_dnn1), dim=1)
        output_cross2 = self.cross2(x_dnn2)
        output_dnn2 = self.dnn2(x_dnn2)
        comb_tensor2 = torch.cat((output_cross2, output_dnn2), dim=1)
        logit1 = self.combination1(comb_tensor1)
        logit2 = self.combination2(comb_tensor2)
        return logit1, logit2


def getOptim(network, optim, lr, l2):
    weight_params = map(lambda a: a[1], filter(lambda p: p[1].requires_grad and 'domain' not in p[0], network.named_parameters()))
    mask_params = map(lambda a: a[1], filter(lambda p: p[1].requires_grad and 'domain' in p[0], network.named_parameters()))

    optim = optim.lower()
    if optim == "sgd":
        return [torch.optim.SGD(weight_params, lr=lr, weight_decay=l2), torch.optim.SGD(mask_params, lr=lr)]
    elif optim == "adam":
        return [torch.optim.Adam(weight_params, lr=lr, weight_decay=l2), torch.optim.Adam(mask_params, lr=lr)]
    else:
        raise ValueError("Invalid optimizer type: {}".format(optim))

def getModel(model: str, opt):
    model = model.lower()
    if model == "deepfm":
        return MaskDeepFM(opt)
    elif model == "dcn":
        return MaskDeepCross(opt)
    elif model == "dnn":
        return MaskDNN(opt)
    else:
        raise ValueError("Invalid model type: {}".format(model))

def sigmoid(x):
    return float(1. / (1. + np.exp(-x)))


