from torch_scatter import scatter_mean, scatter_max, scatter_add
import torch
from torch import nn
import math
import torch.nn.functional as F

import torch


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
            feats = self.node_model(feats, edge_index, edge_attr)

        return feats, edge_attr

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
            edge_attr = torch.cat([feats[r, :], feats[c, :], edge_attr[r, c].unsqueeze(dim=1)], dim=1)
            edge_attr = self.edge_encoder(edge_attr)

        if self.params['use_node_encoder']:
            feats = self.node_encoder(feats)

        feats, _ = self.gnn_model(feats, edge_index, edge_attr)

        x = self.classifier(feats)

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
            self.aggr = lambda out, row, x_size: scatter_add(out, row, dim=0,
                                                             dim_size=x_size)
        if self.gnn_params['aggregator'] == "mean":
            self.aggr = lambda out, row, x_size: scatter_mean(out, row, dim=0,
                                                              dim_size=x_size)
        if self.gnn_params['aggregator'] == "max":
            self.aggr = lambda out, row, x_size: scatter_max(out, row, dim=0,
                                                             dim_size=x_size)

        # init attention mechanism
        if self.gnn_params['attention'] == "dot":
            layers = [MultiHeadDotProduct(embed_dim, gnn_params['num_heads'],
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
        feats, _, _ = self.multi_att(feats, edge_index, edge_attr)
        return feats


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
        self.q_linear = LinearFun(embed_dim, embed_dim)
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
                              edge_index=edge_index, edge_attr=edge_attr)

        # concatenate heads and put through final linear layer
        concat = torch.cat(out, dim=1)
        feats = self.out(concat)

        return feats, edge_index, edge_attr

    def _attention(self, q, k, v, head_dim, dropout=None, edge_index=None,
                   edge_attr=None):
        row, col = edge_index[:, 0], edge_index[:, 1]

        #TODO: Edge attributes
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

        # define mask for nodes that are not connected
        mask = torch.zeros(q.shape[1], q.shape[1])
        mask[row, col] = 1
        mask = mask.unsqueeze(0).to(self.dev)
        scores = scores.masked_fill(mask == 0, -1e4)

        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        scores = scores[:, row, col]  # H x edge_index.shape(0)

        #if edge_attr is not None:
        #    v = torch.cat([v[:, col], edge_attr])

        out = scores.unsqueeze(2).repeat(1, 1, q.shape[2]) * v[:, col]  # H x edge_index.shape(0) x head_dim

        out = [self.aggr(i, row, q.shape[1]) for i in out]

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

    def __init__(self, embed_dim, num_heads, aggr, dev, edge_dim=0, dropout=0.1,
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
            torch.Tensor(self.num_heads, 2 * self.head_dim + self.edge_head_dim, 1))

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
            edge_attr = self.fc_edge(edge_attr).view(e, self.num_heads,
                                                     self.edge_head_dim
                                                     ).transpose(0, 1)

        # num_heads x bs x out_dim
        out = self._attention(feats, edge_index, bs, edge_attr)
        out = out.transpose(0,1).contiguous().view(bs, self.num_heads*self.head_dim)

        if self.bias is not None:
            out += self.bias

        out = self.out(out)

        return out, edge_index, edge_attr.transpose(0, 1).view(e, self.num_heads * self.edge_head_dim)

    def _attention(self, feats, edge_index, bs, edge_attr=None):
        row, col = edge_index[:, 0], edge_index[:, 1]

        if edge_attr is None:
            # num_heads x edge_index.shape(0) x 2 * C
            out = torch.cat(
                [feats[:, row], feats[:, col]],
                dim=2)
        else:
            # num_heads x edge_index.shape(0) x 2 * C + edge_head_dim
            out = torch.cat(
                [feats[:, row], feats[:, col],
                 edge_attr], dim=2)

        alpha = torch.bmm(out, self.att)  # n_heads x edge_ind.shape(0) x 1
        alpha = self.act(self.dropout(alpha))
        alpha = softmax(alpha, row, dim=1, dim_size=bs)

        # H x edge_index.shape(0) x head_dim
        out = alpha * feats[:, col]
        out = scatter_add(out, row, dim=1, dim_size=bs)

        return out


def softmax(src, index, dim, dim_size):
    denom = scatter_add(torch.exp(src), index, dim=dim, dim_size=dim_size)
    if dim == 1:
        out = torch.exp(src) / denom[:, index]
    elif dim == 0:
        out = torch.exp(src) / denom[index]

    return out


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
