import math
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add, scatter_max

from torch_geometric.nn.inits import uniform, kaiming_uniform


class Linear(torch.nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, bias=True):
        super(Linear, self).__init__()
        assert in_channels % groups == 0 and out_channels % groups == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        self.weight = Parameter(
            torch.Tensor(groups, in_channels // groups,
                         out_channels // groups))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform(self.weight, fan=self.weight.size(1), a=math.sqrt(5))
        uniform(self.weight.size(1), self.bias)

    def forward(self, src):
        # Input: [*, in_channels]
        # Output: [*, out_channels]

        if self.groups > 1:
            size = src.size()[:-1]
            src = src.view(-1, self.groups, self.in_channels // self.groups)
            src = src.transpose(0, 1).contiguous()
            out = torch.matmul(src, self.weight)
            out = out.transpose(1, 0).contiguous()
            out = out.view(size + (self.out_channels,))
        else:
            out = torch.matmul(src, self.weight.squeeze(0))

        if self.bias is not None:
            out += self.bias

        return out

    def __repr__(self):  # pragma: no cover
        return '{}({}, {}, groups={})'.format(self.__class__.__name__,
                                              self.in_channels,
                                              self.out_channels, self.groups)


'''def restricted_softmax(src, dim: int = -1, margin: float = 0.):
    src_max = torch.clamp(src.max(dim=dim, keepdim=True)[0], min=0.)
    out = (src - src_max).exp()
    out = out / (out.sum(dim=dim, keepdim=True) + (margin - src_max).exp())
    return out'''


def restricted_softmax(src, index, dim, dim_size, margin: float = 0.):
    src_max = torch.clamp(scatter_max(src.float(), index, dim=dim, dim_size=dim_size)[0], min=0.)
    src = (src - src_max.index_select(dim=dim, index=index)).exp()
    denom = scatter_add(src, index, dim=dim, dim_size=dim_size)
    out = src / (denom + (margin - src_max).exp()).index_select(dim, index)

    return out


class Attention(torch.nn.Module):
    def __init__(self, dropout=0):
        super(Attention, self).__init__()
        self.dropout = dropout

    def forward(self, query, key, value):
        return self.compute_attention(query, key, value)

    def compute_attention(self, query, key, value, index, num_nodes):
        # query_entries = key_entries = value_entries = num_edges
        # query: [heads, query_entries, dim_k]
        # key: [heads, key_entries, dim_k]
        # value: [heads, value_entries, dim_v]
        # Output: [heads, query_edges, dim_v]

        assert query.dim() == key.dim() == value.dim() >= 2
        assert query.size(-1) == key.size(-1)
        assert key.size(-2) == value.size(-2)

        # Score: [query_entries, key_entries]
        score = torch.matmul(query.unsqueeze(-2), key.unsqueeze(-1))
        score = score / math.sqrt(key.size(-1))
        score = restricted_softmax(score, index, dim=1, dim_size=num_nodes)
        score = F.dropout(score, p=self.dropout, training=self.training)

        return torch.matmul(score, value.unsqueeze(-2)).squeeze()

    def __repr__(self):  # pragma: no cover
        return '{}(dropout={})'.format(self.__class__.__name__, self.dropout)


class MultiHead(Attention):
    def __init__(self, in_channels, out_channels, heads=1, groups=1, dropout=0,
                 bias=True):
        super(MultiHead, self).__init__(dropout)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_dim = int(out_channels / heads)
        self.num_heads = heads
        self.groups = groups
        self.bias = bias

        assert in_channels % heads == 0 and out_channels % heads == 0
        assert in_channels % groups == 0 and out_channels % groups == 0
        assert max(groups, self.num_heads) % min(groups, self.num_heads) == 0

        self.lin_q = Linear(in_channels, out_channels, groups, bias)
        self.lin_k = Linear(in_channels, out_channels, groups, bias)
        self.lin_v = Linear(in_channels, out_channels, groups, bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_q.reset_parameters()
        self.lin_k.reset_parameters()
        self.lin_v.reset_parameters()

    def forward(self, query, key, value, index, num_nodes):
        # query: [num_edges, in_channels]
        # key: [num_edges, in_channels]
        # value: [num_edges, in_channels]
        # Output: [num_nodes, out_channels]

        assert query.dim() == key.dim() == value.dim() >= 2
        assert query.size(-1) == key.size(-1) == value.size(-1)
        assert key.size(-2) == value.size(-2)

        num_edges = query.shape[0]

        key = self.lin_k(key).view(num_edges, self.num_heads,
                                   self.head_dim).transpose(0, 1)
        query = self.lin_q(query).view(num_edges, self.num_heads,
                                       self.head_dim).transpose(0, 1)
        value = self.lin_v(value).view(num_edges, self.num_heads,
                                       self.head_dim).transpose(0, 1)

        # Output: [heads, num_edges, out_channels // heads]
        out = self.compute_attention(query, key, value, index, num_nodes)
        # Output: [num_edges, heads, out_channels // heads]
        out = out.transpose(0, 1).contiguous()
        # Output: [num_edges, out_channels]
        out = out.view(num_edges, self.out_channels)

        return out

    def __repr__(self):  # pragma: no cover
        return '{}({}, {}, heads={}, groups={}, dropout={}, bias={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.heads, self.groups, self.dropout, self.bias)


class AttentionLayerDot(MessagePassing):

    def __init__(self, channels: int, heads: int = 1, groups: int = 1,
                dropout: float = 0., gcn_norm: int = 0, mult_edge_weight: int = 1,
                add_self_loops: bool = True, bias: bool = True, **kwargs):
        super(AttentionLayerDot, self).__init__(aggr='add', node_dim=0,
                                                **kwargs)
        print("AttentionLayerDot")
        self.bias = bias
        self.add_self_loops = add_self_loops
        self.gcn_norm = gcn_norm
        self.mult_edge_weight = mult_edge_weight
        self.num_nodes = 0

        self.multi_head = MultiHead(channels, channels, heads, groups, dropout,
                                    bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.multi_head.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_weight: OptTensor = None) -> Tensor:
        r"""
        Args:
            x: The input node features of shape :obj:`[num_nodes, channels]`.
            edge_index: Tensor of shape :obj:`[num_edges, 2]`.
            edge_weight: Tensor of shape :obj:`[num_edges, 2]`.
        """
        self.num_nodes = x.shape[0]

        if x.dim() != 2:
            raise ValueError('Feature shape must be [num_nodes, channels].')

        if edge_index.shape[0] != 2:
            edge_index = edge_index.transpose(-2, -1)

        if self.gcn_norm:
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, dtype=x.dtype)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight,
                              size=None), edge_index, edge_weight

    def message(self, x_i: Tensor, x_j: Tensor, edge_weight: Tensor,
                index: Tensor) -> Tensor:
        out = self.multi_head(x_i, x_j, x_j, index, self.num_nodes)  # [num_edges, channels]
        if self.mult_edge_weight:
            return edge_weight.view(-1, 1) * out.squeeze(1)
        else:
            return out.squeeze(1)

    def __repr__(self):
        return '{}({}, heads={}, groups={})'.format(
            self.__class__.__name__, self.multi_head.in_channels,
            self.multi_head.heads, self.multi_head.groups)


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    fill_value = 2. if improved else 1.

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
    elif len(edge_weight.size()) != 1:
        edge_weight = edge_weight.squeeze()

    if edge_index.shape[0] != 2:
            edge_index = edge_index.transpose(-2, -1)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
