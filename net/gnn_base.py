from torch_scatter import scatter_mean, scatter_max, scatter_add
import torch
from torch import nn
import math
import torch.nn.functional as F
import numpy as np
import torch
from torch_geometric import nn as nn_geo
import logging
from .dot_attention_pygeo import AttentionLayerDot, gcn_norm
from .utils import *
from .attentions import MultiHeadDotProduct, MultiHeadMLP

logger = logging.getLogger('GNNReID.GNNModule')


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
        num_classes = params['classifier']['num_classes']
        self.dev = dev
        self.params = params
        self.node_endocer_params = params['node_encoder']
        self.edge_encoder_params = params['edge_encoder']
        self.edge_params = params['edge']
        self.gnn_params = params['gnn']

        if self.params['use_edge_encoder']:
            self.edge_encoder = MLP(2 * embed_dim + 1,
                                    **self.edge_encoder_params)

        if self.params['use_node_encoder']:
            self.node_encoder = MLP(embed_dim, **self.node_endocer_params)

        self.gnn_model = self._build_GNN_Net(embed_dim=embed_dim)

        # classifier
        self.neck = params['classifier']['neck']
        if self.neck:
            self.bottleneck = nn.BatchNorm1d(embed_dim)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.fc = nn.Linear(embed_dim, num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.fc.apply(weights_init_classifier)
        else:
            self.fc = nn.Linear(embed_dim, num_classes)

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

        # init aggregator
        if self.gnn_params['aggregator'] == "add":
            self.aggr = lambda out, row, dim, x_size: scatter_add(out, row,
                                                                  dim=dim,
                                                                  dim_size=x_size)
        if self.gnn_params['aggregator'] == "mean":
            self.aggr = lambda out, row, dim, x_size: scatter_mean(out,
                                                                   row,
                                                                   dim=dim,
                                                                   dim_size=x_size)
        if self.gnn_params['aggregator'] == "max":
            self.aggr = lambda out, row, dim, x_size: scatter_max(out, row,
                                                                  dim=dim,
                                                                  dim_size=x_size)

        layers = [DotAttentionLayer(embed_dim, self.aggr, self.dev,
                                    edge_dim, self.gnn_params) for _
                  in range(self.gnn_params['num_layers'])]

        gnn = Sequential(*layers)

        return MetaLayer(edge_model=edge_model, node_model=gnn)

    def forward(self, feats, edge_index, edge_attr=None, output_option='norm'):
        r, c = edge_index[:, 0], edge_index[:, 1]

        if self.params['use_edge_encoder']:
            edge_attr = torch.cat(
                [feats[r, :], feats[c, :], edge_attr.unsqueeze(dim=1)], dim=1)
            edge_attr = self.edge_encoder(edge_attr)

        if self.params['use_node_encoder']:
            feats = self.node_encoder(feats)

        feats, _, _ = self.gnn_model(feats, edge_index, edge_attr)

        if self.neck:
            features = self.bottleneck(feats)
        else:
            features = feats

        x = self.fc(features)

        if output_option == 'norm':
            return x, feats
        elif output_option == 'plain':
            return x, F.normalize(feats, p=2, dim=1)
        elif output_option == 'neck' and self.neck:
            return x, features
        elif output_option == 'neck' and not self.neck:
            print("Output option neck only avaiable if bottleneck (neck) is "
                  "enabeled - giving back x and fc7")
            return x, feats

        return x, feats


class DotAttentionLayer(nn.Module):
    def __init__(self, embed_dim, aggr, dev, edge_dim, params, d_hid=None):
        super(DotAttentionLayer, self).__init__()
        num_heads = params['num_heads']
        self.res1 = params['res1']
        self.res2 = params['res2']

        # try AttentionLayerDot
        if params['attention'] == "dot":
            self.att = MultiHeadDotProduct(embed_dim, num_heads, aggr,
                                           edge_dim)
        elif params['attention'] == "dot_pygeo":
            self.att = AttentionLayerDot(embed_dim, num_heads)

        elif params['attention'] == "mlp":
            self.att = MultiHeadMLP(embed_dim, num_heads, aggr, dev, edge_dim)

        else:
            print(
                "Invalid attention option {}. Please choose from dot/dot_pygeo/mlp".format(
                    self.params['attention']))

        d_hid = 4 * embed_dim if d_hid is None else d_hid
        self.mlp = params['mlp']

        self.linear1 = nn.Linear(embed_dim, d_hid) if params['mlp'] else None
        self.dropout = nn.Dropout(params['dropout_mlp'])
        self.linear2 = nn.Linear(d_hid, embed_dim) if params['mlp'] else None

        self.norm1 = LayerNorm(embed_dim) if params['norm1'] else None
        self.norm2 = LayerNorm(embed_dim) if params['norm2'] else None
        self.dropout1 = nn.Dropout(params['dropout_1'])
        self.dropout2 = nn.Dropout(params['dropout_2'])

        self.act = F.relu

    def forward(self, feats, egde_index, edge_attr):
        feats2, egde_index, edge_attr = self.att(feats, egde_index, edge_attr)
        feats2 = self.dropout1(feats2)
        feats = feats + feats2 if self.res1 else feats2
        feats = self.norm1(feats) if self.norm1 is not None else feats

        if self.mlp:
            feats2 = self.linear2(self.dropout(self.act(self.linear1(feats))))
        else:
            feats2 = feats

        feats2 = self.dropout2(feats2)
        feats = feats + feats2 if self.res2 else feats2
        feats = self.norm2(feats) if self.norm2 is not None else feats

        return feats, egde_index, edge_attr


