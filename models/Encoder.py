from json import encoder
import torch
import torch.nn as nn
from models.Decoder import dict2obj
import torch.nn.functional as F

__all__ = ('Encoder_HighWay', 'Transformer')


class SP(nn.Module):
    """SP layer implementation

    Args:
        num_clusters : int
            The number of pseudo regions
        dim : int
            Dimension of pseudo regions
        alpha : float
            Parameter of initialization. Larger value is harder assignment.
        normalize_input : bool
            If true, pseudo regions-wise L2 normalization is applied to input.
    """

    def __init__(self, num_regions=64, dim=128, alpha=100.0, normalize_input=True):
        super().__init__()
        self.num_regions = num_regions
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_regions, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_regions, dim))
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, grids):

        N, frame, C = grids.shape

        grids = (
            grids.view(N, frame, 1, -1).permute(0, 3, 1, 2).contiguous()
        )  # N dim grid 1

        if self.normalize_input:
            grids = F.normalize(grids, p=2, dim=1)  # across descriptor dim

        soft_assign = self.conv(grids).view(N, self.num_regions, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = grids.view(N, C, -1)

        residual = x_flatten.expand(self.num_regions, -1, -1, -1).permute(
            1, 0, 2, 3
        ).contiguous() - self.centroids.expand(x_flatten.size(-1), -1, -1).permute(
            1, 2, 0
        ).contiguous().unsqueeze(
            0
        )

        residual *= soft_assign.unsqueeze(2)
        p = residual.sum(dim=-1)

        p = F.normalize(p, p=2, dim=2)  # intra-normalization
        p = p.view(grids.size(0), -1)
        p = F.normalize(p, p=2, dim=1)  # L2 normalize

        return p.reshape(N, self.num_regions, -1)


class HighWay(nn.Module):
    def __init__(self, hidden_size, with_gate=True):
        super(HighWay, self).__init__()
        self.with_gate = with_gate
        self.w1 = nn.Linear(hidden_size, hidden_size)
        if self.with_gate:
            self.w2 = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        # self._init_weights()

    def forward(self, x):
        '''
        x : [Tensor] (bs frames/clips dim)
        '''
        y = self.tanh(self.w1(x))
        if self.with_gate:
            gate = torch.sigmoid(self.w2(x))
            return gate * x + (1 - gate) * y
        else:
            return x + y


class MultipleStreams(nn.Module):
    def __init__(self, opt, module_func, is_rnn=False):
        super(MultipleStreams, self).__init__()
        self.encoders = []

        modality = opt['modality'].lower()
        for char in modality:
            input_dim = opt.get('dim_' + char, None)
            output_dim = opt.get('dim_hidden', 512)
            dropout = opt.get('encoder_dropout', 0.5)
            assert (
                input_dim is not None
            ), 'The modality is {}, but dim_{} can not be found in opt'.format(
                modality, char
            )

            module = module_func(input_dim * 2, output_dim, dropout)
            self.add_module("Encoder_%s" % char.upper(), module)
            self.encoders.append(module)

        # self.fc1 = nn.Linear(2048, 512)
        # self.fc2 = nn.Linear(2048, 512)
        # self.fc = [self.fc1, self.fc2]

        self.num_feats = len(modality)
        self.is_rnn = is_rnn
        self.summary = SP(num_regions=5, dim=4096)

    def forward(self, input_feats):
        assert self.num_feats == len(input_feats)
        if not self.is_rnn:

            # baseline w/ descriptions
            # appmot = torch.cat([input_feats[0], input_feats[1]], dim=2)
            # description = self.summary(appmot)
            # input_feats = [appmot] + [description]

            # baseline
            input_feats = [torch.cat([input_feats[0], input_feats[1]], dim=2)]
            # encoder_descriptions = [encoder(summary(feats)) for encoder, summary, feats in zip(self.encoders, self.summary, input_feats)]
            encoder_ouputs = [
                encocder(feats) for encocder, feats in zip(self.encoders, input_feats)
            ]
            # encoder_ouputs_ = [encocder(feats) for encocder, feats in zip(self.encoders, input_feats)] # bs frame/clip dim
            # encoder_ouputs = encoder_ouputs_ + encoder_descriptions
            encoder_hiddens = [item.mean(1) for item in encoder_ouputs]  # bs dim
        else:
            pass
            # TODO

        if getattr(self, 'subsequent_processing', None) is not None:
            return self.subsequent_processing(encoder_ouputs, encoder_hiddens)

        return encoder_ouputs, encoder_hiddens


class Encoder_HighWay(MultipleStreams):
    def __init__(self, opt):
        with_gate = opt.get('gate', True)
        module_func = lambda x, y, z: nn.Sequential(
            nn.Linear(x, y), HighWay(y, with_gate), nn.Dropout(z)
        )
        super(Encoder_HighWay, self).__init__(opt, module_func)


class Transformer(nn.Module):
    def __init__(self, opt):
        super(Transformer, self).__init__()
        self.opt = opt
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.fc = nn.Linear(4096, 512)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(512)
        self.relu = nn.ReLU()
        self.appmot_seq = nn.Sequential(
            self.fc,
            # self.relu,
            # self.dropout,
            # self.layer_norm,
            # nn.Linear(2048, 512), # 效果很差,弃
            self.encoder,
        )
        self.description_seq = nn.Sequential(
            self.fc,
            # self.relu,
            # self.dropout,
            # self.layer_norm,
            # nn.Linear(2048, 512), # 效果很差,弃
            self.encoder,
        )
        self.summary = SP(num_regions=self.opt['cluster_numbers'], dim=4096)

    def forward(self, input_feats):
        appmot = torch.cat([input_feats[0], input_feats[1]], dim=-1)

        if self.opt['cluster']:
            # print("****************running extension model...****************")
            description = self.summary(appmot)

            appmot = self.appmot_seq(appmot)
            description = self.description_seq(description)

            encoder_ouputs = [torch.cat([appmot, description], dim=1)]
            encoder_hiddens = [item.mean(1) for item in encoder_ouputs]
        else:
            # print("****************running baseline model...****************")
            encoder_ouputs = [self.appmot_seq(appmot)]
            encoder_hiddens = [item.mean(1) for item in encoder_ouputs]
        return encoder_ouputs, encoder_hiddens
