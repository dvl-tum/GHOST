from .utils import *
from torch import nn
import torch
import math


class MultiHeadDotProduct(nn.Module):
    """
    Multi head attention like in transformers
    embed_dim: dimension of input embedding
    nhead: number of attention heads
    """

    def __init__(self, embed_dim, nhead, aggr, edge_dim, dropout=0.1):
        super(MultiHeadDotProduct, self).__init__()
        print("MultiHeadDotProduct")
        self.embed_dim = embed_dim
        self.hdim = embed_dim // nhead
        self.nhead = nhead
        self.aggr = aggr

        # FC Layers for input
        self.q_linear = nn.Linear(embed_dim + edge_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim + edge_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim + edge_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        # fc layer for concatenated output
        self.out = nn.Linear(embed_dim, embed_dim)

        self.reset_parameters()

    def forward(self, feats: torch.tensor, edge_index: torch.tensor,
                edge_attr: torch.tensor):
        q = k = v = feats
        bs = q.size(0)

        # FC layer and split into heads --> h * bs * embed_dim
        k = self.k_linear(k).view(bs, self.nhead, self.hdim).transpose(0, 1)
        q = self.q_linear(q).view(bs, self.nhead, self.hdim).transpose(0, 1)
        v = self.v_linear(v).view(bs, self.nhead, self.hdim).transpose(0, 1)

        # perform multi-head attention
        feats = self._attention(q, k, v, edge_index, edge_attr, bs)

        # concatenate heads and put through final linear layer
        feats = feats.transpose(0, 1).contiguous().view(
            bs, self.nhead * self.hdim)
        feats = self.out(feats)

        return feats, edge_index, edge_attr

    def _attention(self, q, k, v, edge_index=None, edge_attr=None, bs=None):
        r, c, e = edge_index[:, 0], edge_index[:, 1], edge_index.shape[0]

        scores = torch.matmul(
            q.index_select(1, c).unsqueeze(dim=-2),
            k.index_select(1, r).unsqueeze(dim=-1))
        scores = scores.view(self.nhead, e, 1) / math.sqrt(self.hdim)
        scores = softmax(scores, c, 1, bs)
        scores = self.dropout(scores)

        out = scores * v.index_select(1, r)  # H x e x hdim
        out = self.aggr(out, c, 1, bs)  # H x bs x hdim

        return out

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.constant_(self.q_linear.bias, 0.)

        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.constant_(self.v_linear.bias, 0.)

        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.constant_(self.k_linear.bias, 0.)

        #nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, 0.)


class MultiHeadMLP(nn.Module):
    """
        Multi head attention like in GAT
        embed_dim: dimension of input embedding
        nhead: number of attention heads
        """

    def __init__(self, embed_dim, nhead, aggr, edge_dim=0,
                 dropout=0.1, bias=True, concat=True):
        super(MultiHeadMLP, self).__init__()
        self.embed_dim = embed_dim
        self.hdim = embed_dim // nhead
        self.nhead = nhead
        self.aggr = aggr
        self.edge_dim = edge_dim

        # from embed dim to nhead * hdim = embed_dim
        self.fc = LinearFun(embed_dim, embed_dim)

        if edge_dim != 0:
            self.edge_hdim = edge_dim // nhead
            self.fc_edge = LinearFun(edge_dim, edge_dim, bias=False)
        else:
            self.edge_hdim = 0

        self.att = torch.nn.Parameter(
            torch.Tensor(self.nhead,
                         2 * self.hdim + self.edge_hdim, 1))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(nhead * self.hdim))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(self.hdim))
        else:
            self.register_parameter('bias', None)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU()
        self.out = LinearFun(embed_dim, embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()
        self.out.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self, feats: torch.tensor, edge_index: torch.tensor,
                edge_attr: torch.tensor = None):
        bs, e = feats.shape[0], edge_attr.shape[0]

        # nhead x bs x hdim
        feats = self.fc(feats).view(bs, self.nhead, self.hdim).transpose(0, 1)

        if self.edge_hdim:
            # nhead x e x edge_hdim
            edge_attr_att = self.fc_edge(edge_attr).view(e, self.nhead,
                                                         self.edge_hdim)
            edge_attr_att = edge_attr_att.transpose(0, 1)
        else:
            edge_attr_att = edge_attr

        # nhead x bs x out_dim
        out = self._attention(feats, edge_index, bs, edge_attr_att)
        out = out.transpose(0, 1).contiguous().view(bs, self.nhead * self.hdim)

        if self.bias is not None:
            out += self.bias

        out = self.out(out)

        return out, edge_index, edge_attr

    def _attention(self, feats, edge_index, bs, edge_attr=None):
        r, c = edge_index[:, 0], edge_index[:, 1]

        if not self.edge_hdim:  # nhead x edge_index.shape(0) x 2 * C
            out = torch.cat(
                [feats.index_select(1, c), feats.index_select(1, r)], dim=2)
        else:  # nhead x edge_index.shape(0) x 2 * C + edge_hdim
            out = torch.cat(
                [feats.index_select(1, c), feats.index_select(1, r),
                 edge_attr], dim=2)

        alpha = torch.bmm(out, self.att)  # n_heads x edge_ind.shape(0) x 1
        alpha = self.dropout(softmax(self.act(alpha), c, dim=1, dim_size=bs))

        # H x edge_index.shape(0) x hdim
        feats = alpha * feats.index_select(1, r)
        feats = self.aggr(feats, c, 1, bs)

        return feats
