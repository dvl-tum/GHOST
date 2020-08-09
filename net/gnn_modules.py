from torch_scatter import scatter_mean, scatter_max, scatter_add
import torch
from torch import nn
import math
import torch.nn.functional as F
import numpy as np
import torch
from torch_geometric import nn as nn_geo


class MetaLayer(torch.nn.Module):
    """
        Core Message Passing Network Class. Extracted from torch_geometric, with minor modifications.
        (https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/meta.py)
    """

    def __init__(self, edge_model=None, node_model=None):
        super(MetaLayer, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, feats, edge_index, edge_attr=None):
        """"""

        r, c = edge_index[:, 0], edge_index[:, 1]

        if self.edge_model is not None:
            edge_attr = torch.cat([feats[r], feats[c], edge_attr], dim=1)
            edge_attr = self.edge_model(edge_attr)

        if self.node_model is not None:
            feats, edge_index, edge_attr = self.node_model(feats, edge_index,
                                                           edge_attr)

        return feats, edge_index, edge_attr

    def __repr__(self):
        return ('{}(\n'
                '    edge_model={},\n'
                '    node_model={},\n'
                '    global_model={}\n'
                ')').format(self.__class__.__name__, self.edge_model,
                            self.node_model, self.global_model)


class GNNReID(nn.Module):
    def __init__(self, dev, params: dict = None, embed_dim: int = 2048):
        super(GNNReID, self).__init__()

        self.dev = dev
        self.params = params
        self.node_endocer_params = params['node_encoder']
        self.edge_encoder_params = params['edge_encoder']
        self.edge_params = params['edge']
        self.gnn_params = params['gnn']
        self.class_params = params['classifier']

        if self.params['use_edge_encoder']:
            self.edge_encoder = MLP(2 * embed_dim + 1,
                                    **self.edge_encoder_params)

        if self.params['use_node_encoder']:
            self.node_encoder = MLP(embed_dim, **self.node_endocer_params)

        self.gnn_model = self._build_GNN_Net(embed_dim=embed_dim)

        self.classifier = MLP(embed_dim, **self.class_params)

    def _build_GNN_Net(self, embed_dim: int = 2048):

        if self.params['use_edge_model']:
            edge_model = MLP(
                2 * embed_dim + self.edge_encoder_params['fc_dims'][-1],
                **self.edge_params)
            edge_dim = self.edge_params['fc_dims'][-1]

        elif self.params['use_edge_encoder']:
            edge_model = None
            edge_dim = self.edge_encoder_params['fc_dims'][-1]

        else:
            edge_model = None
            edge_dim = 0

        gnn = GNN(self.dev, self.gnn_params, embed_dim, edge_dim)

        return MetaLayer(edge_model=edge_model, node_model=gnn)

    def forward(self, feats, edge_index, edge_attr=None):

        if self.params['use_edge_encoder']:
            r, c = edge_index[:, 0], edge_index[:, 1]
            edge_attr = torch.cat(
                [feats[r, :], feats[c, :], edge_attr[r, c].unsqueeze(dim=1)],
                dim=1)
            edge_attr = self.edge_encoder(edge_attr)

        if self.params['use_node_encoder']:
            feats = self.node_encoder(feats)

        feats, _, _ = self.gnn_model(feats, edge_index, edge_attr)

        if (torch.isnan(feats) == True).any().item():
            print(106)
            print(feats)

        x = self.classifier(feats)
        if (torch.isnan(x) == True).any().item():
            print(111)
            print(x)

        return x, feats


class GNN(nn.Module):
    """
    Main GNN Class for graph neural network based person re-identification
    1. Encoder: optional encoding of node features
    2. GNN: graph neural network to pass messages between the nodes (samples)
    3. Classifier: Classifier layer after the message passing

    Idea: probably classify after each message passing step and aggregate or
    concatenate in the end
    """

    def __init__(self, dev, gnn_params: dict = None, embed_dim: int = 2048,
                 edge_dim: int = 0):
        super(GNN, self).__init__()
        self.gnn_params = gnn_params
        self.dev = dev

        # init aggregator
        if self.gnn_params['aggregator'] == "add":
            self.aggr = lambda out, row, dim, x_size: scatter_add(out, row,
                                                                  dim=dim,
                                                                  dim_size=x_size)
        if self.gnn_params['aggregator'] == "mean":
            self.aggr = lambda out, row, dim, x_size: scatter_mean(out, row,
                                                                   dim=dim,
                                                                   dim_size=x_size)
        if self.gnn_params['aggregator'] == "max":
            self.aggr = lambda out, row, dim, x_size: scatter_max(out, row,
                                                                  dim=dim,
                                                                  dim_size=x_size)

        # init attention mechanism
        if self.gnn_params['attention'] == "dot":
            layers = [DotAttentionLayer(embed_dim, gnn_params['num_heads'],
                                          self.aggr, self.dev, edge_dim) for _
                      in range(self.gnn_params['num_layers'])]
            self.multi_att = Sequential(*layers)

        elif self.gnn_params['attention'] == "mlp":
            layers = [
                MultiHeadMLP(embed_dim, gnn_params['num_heads'], self.aggr,
                             self.dev, edge_dim) for _ in
                range(self.gnn_params['num_layers'])]
            self.multi_att = Sequential(*layers)

        else:
            print(
                "Invalid attention option {}. Please choose from dot/mlp".format(
                    self.gnn_params['attention']))

    def forward(self, feats, edge_index, edge_attr=None):
        feats, edge_index, edge_attr = self.multi_att(feats, edge_index,
                                                      edge_attr)
        return feats, edge_index, edge_attr


class DotAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, aggr, dev, edge_dim, dropout=0.4):
        super(DotAttentionLayer, self).__init__()
        self.multi_att = MultiHeadDotProduct(embed_dim, num_heads, aggr, dev,
                                             edge_dim)

        self.layer_norm1 = LayerNorm(norm_shape=embed_dim) #nn_geo.LayerNorm(embed_dim) #LayerNorm(norm_shape=embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm2 = LayerNorm(norm_shape=embed_dim) #nn_geo.LayerNorm(embed_dim) #LayerNorm(norm_shape=embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = MLP(embed_dim, fc_dims=[embed_dim*4, embed_dim])

    def forward(self, feats, egde_index, edge_attr):

        feats2 = self.layer_norm1(feats)
        feats2, egde_index, edge_attr = self.multi_att(feats2, egde_index,
                                                       edge_attr)
        feats = feats + self.dropout1(feats2)
        feats2 = self.layer_norm2(feats)
        feats = feats + self.dropout2(self.fc(feats2))

        return feats, egde_index, edge_attr


class MultiHeadDotProduct(nn.Module):
    """
    Multi head attention like in transformers
    embed_dim: dimension of input embedding
    num_heads: number of attetion heads
    """

    def __init__(self, embed_dim, num_heads, aggr, dev, edge_dim, dropout=0.1):
        super(MultiHeadDotProduct, self).__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.aggr = aggr
        self.dev = dev

        # FC Layers for input
        self.q_linear = LinearFun(embed_dim+edge_dim, embed_dim+edge_dim)
        self.v_linear = LinearFun(embed_dim, embed_dim)
        self.k_linear = LinearFun(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        # fc layer for concatenated output
        self.out = nn.Linear(embed_dim, embed_dim)

        self.reset_parameters()

    def forward(self, feats: torch.tensor, edge_index: torch.tensor,
                edge_attr: torch.tensor):
        q = k = v = feats
        bs = q.size(0)

        # FC layer and split into heads
        k = self.k_linear(k)
        k = k.view(bs, self.num_heads, self.head_dim)
        q = self.q_linear(q).view(bs, self.num_heads, self.head_dim)
        v = self.v_linear(v).view(bs, self.num_heads, self.head_dim)

        # h * bs * embed_dim
        k = k.transpose(0, 1)
        q = q.transpose(0, 1)
        v = v.transpose(0, 1)

        # perform multihead attention
        out = self._attention(q, k, v, self.head_dim, dropout=self.dropout,
                              edge_index=edge_index, edge_attr=edge_attr,
                              bs=bs)

        # concatenate heads and put through final linear layer
        out = out.transpose(0, 1).contiguous().view(bs,
                                                    self.num_heads * self.head_dim)
        if (torch.isnan(out) == True).any().item():
            print(219)
            print(out)

        feats = self.out(out)

        if (torch.isnan(feats) == True).any().item():
            print(223)
            print(feats)

        return feats, edge_index, edge_attr

    def _attention(self, q, k, v, head_dim, dropout=None, edge_index=None,
                   edge_attr=None, bs=None):
        row, col = edge_index[:, 0], edge_index[:, 1]
        e = edge_index.shape[0]

        # TODO: Edge attributes
        # # H x edge_index.shape(0)
        scores = torch.bmm(
            q.index_select(1, col).view(self.num_heads * e, head_dim).unsqueeze(dim=1),
            k.index_select(1, row).view(self.num_heads * e, head_dim).unsqueeze(
                dim=2)).view(self.num_heads, e, 1) / math.sqrt(head_dim)

        if (torch.isnan(scores) == True).any().item():
            print(243)
            print(scores)
            print(torch.max(scores, dim=-1))

        scores_before = scores
        scores = softmax(scores, row, 1, bs)

        if (torch.isnan(scores) == True).any().item():
            print(249)
            print(scores)
            print(torch.max(scores_before, dim=-1))
            print(torch.max(scores, dim=-1))

        if dropout is not None:
            scores = self.dropout(scores)

        if (torch.isnan(scores) == True).any().item():
            print(256)
            print(scores)

        # H x edge_index.shape(0) x head_dim
        out = scores * v.index_select(1, row)
        out = self.aggr(out, col, 1, bs)

        if (torch.isnan(out) == True).any().item():
            print(264)
            print(out)

        return out

    def reset_parameters(self):
        self.q_linear.reset_parameters()
        self.k_linear.reset_parameters()
        self.v_linear.reset_parameters()

        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, 0.)


class MultiHeadMLP(nn.Module):
    """
        Multi head attention like in GAT
        embed_dim: dimension of input embedding
        num_heads: number of attetion heads
        """

    def __init__(self, embed_dim, num_heads, aggr, dev, edge_dim=0,
                 dropout=0.1,
                 act='leaky', bias=True, concat=True):
        super(MultiHeadMLP, self).__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.aggr = aggr
        self.edge_dim = edge_dim

        # from embed dim to num_heads * head_dim = embed_dim
        self.fc = LinearFun(embed_dim, embed_dim, bias=False)

        if edge_dim != 0:
            self.edge_head_dim = edge_dim // num_heads
            self.fc_edge = LinearFun(edge_dim, edge_dim, bias=False)

        self.att = torch.nn.Parameter(
            torch.Tensor(self.num_heads,
                         2 * self.head_dim + self.edge_head_dim, 1))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_heads * self.head_dim))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(self.head_dim))
        else:
            self.register_parameter('bias', None)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

        assert act in ['leaky', 'relu'], 'Choose act out of leaky and relu'

        if act == 'leaky':
            self.act = nn.LeakyReLU()
        else:
            self.act = nn.ReLU()

        self.out = LinearFun(embed_dim, embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()
        self.out.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self, feats: torch.tensor, edge_index: torch.tensor,
                edge_attr: torch.tensor = None):
        bs = feats.shape[0]

        # num_heads x bs x head_dim
        feats = self.fc(feats).view(bs, self.num_heads,
                                    self.head_dim).transpose(0, 1)
        if edge_attr != None:
            e = edge_attr.shape[0]
            # num_heads x e x edge_head_dim
            edge_attr_att = self.fc_edge(edge_attr).view(e, self.num_heads,
                                                         self.edge_head_dim
                                                         ).transpose(0, 1)

        # num_heads x bs x out_dim
        out = self._attention(feats, edge_index, bs, edge_attr_att)
        out = out.transpose(0, 1).contiguous().view(bs,
                                                    self.num_heads * self.head_dim)

        if self.bias is not None:
            out += self.bias

        out = self.out(out)

        return out, edge_index, edge_attr

    def _attention(self, feats, edge_index, bs, edge_attr=None):
        row, col = edge_index[:, 0], edge_index[:, 1]

        if edge_attr is None:
            # num_heads x edge_index.shape(0) x 2 * C
            out = torch.cat(
                [feats.index_select(1, col), feats.index_select(1, row)],
                dim=2)
        else:
            # num_heads x edge_index.shape(0) x 2 * C + edge_head_dim
            out = torch.cat(
                [feats.index_select(1, col), feats.index_select(1, row),
                 edge_attr], dim=2)

        alpha = torch.bmm(out, self.att)  # n_heads x edge_ind.shape(0) x 1
        alpha = self.dropout(softmax(self.act(alpha), col, dim=1, dim_size=bs))

        # H x edge_index.shape(0) x head_dim
        out = alpha * feats.index_select(1, row)
        out = self.aggr(out, col, 1, bs)

        return out


def softmax(src, index, dim, dim_size):
    src = src - scatter_max(src.float(), index, dim=dim, dim_size=dim_size)[0].index_select(dim, index)
    denom = scatter_add(torch.exp(src), index, dim=dim, dim_size=dim_size)
    out = torch.exp(src) / denom.index_select(dim, index)

    return out


class LayerNorm(nn.Module):
    def __init__(self, norm_shape=None, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.affine = affine

        if isinstance(norm_shape, int):
            norm_shape = (norm_shape,)
        self.norm_shape = norm_shape

        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(*self.norm_shape))
            self.bias = nn.Parameter(torch.Tensor(*self.norm_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def forward(self, x):
        init_shape = [x.shape[i] for i in range(len(x.shape))]

        dims = len(self.norm_shape)
        shape = [x.shape[i] for i in range(len(x.shape) - dims)] + [
            int(np.prod(list(self.norm_shape)))]
        x = x.view(shape)

        x = (x - x.mean(dim=-1, keepdim=True)) / (
                    x.std(dim=-1, keepdim=True) + self.eps)

        x = x.view(init_shape)

        if self.affine:
            x *= self.weight
            x += self.bias

        return x

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)


class LinearFun(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(LinearFun, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.bias = bias

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.bias:
            nn.init.constant_(self.linear.bias, 0.)

    def forward(self, x):
        return self.linear(x)


class Sequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class MLP(nn.Module):
    """
    From Guillem
    """

    def __init__(self, input_dim, classifier=False, fc_dims=None,
                 dropout_p=0.4, use_batchnorm=False):

        super(MLP, self).__init__()
        self.classifier = classifier

        assert isinstance(fc_dims, (list,
                                    tuple)), 'fc_dims must be either a list or a tuple, but got {}'.format(
            type(fc_dims))

        layers = []
        self.input_dim = input_dim
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            if use_batchnorm and dim != 1:
                layers.append(nn.BatchNorm1d(dim))

            if dim != 1:
                layers.append(nn.ReLU(inplace=True))

            if dropout_p is not None and dim != 1:
                layers.append(nn.Dropout(p=dropout_p))

            input_dim = dim

        self.fc_layers = nn.Sequential(*layers)

    def reset_parameters(self):
        '''if self.classifier:
            model.bottleneck.bias.requires_grad_(False)'''
        pass

    def forward(self, x):
        return self.fc_layers(x)


class Classifier(nn.Module):
    def __init__(self, input_dim, out_dim, dropout_p=0, neck=False):
        super(MLP, self).__init__()

        self.fc = nn.Linear(input_dim, out_dim)
        if neck:
            self.neck = nn.BatchNorm1d(out_dim)

        if dropout_p != 0:
            self.dropout = nn.Dropout(p=dropout_p)

    def reset_parameters(self):
        self.neck.bias.requires_grad_(False)
        self.neck.bottleneck.apply(weights_init_kaiming)
        self.fc.apply(weights_init_classifier)

    def forward(self, input, output_option):

        if output_option == 'plain':
            x = F.normalize(input, p=2, dim=1)
            return x
        elif output_option == 'neck':
            x = self.neck(input)

        return self.fc(x)


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
