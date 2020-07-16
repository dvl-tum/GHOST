import torch.nn as nn
from torch import Tensor
import copy
from torch.nn.modules.container import ModuleList
import torch.nn.functional as F
import torch
import math


class Labeler(nn.Module):
    def __init__(self, total_classes, device='cuda:0', sz_embed=2048, proxies=0):
        super(Labeler, self).__init__()
        self.m = total_classes
        self.device = device
        self.prox = proxies
        if self.prox:
            # self.proxies = torch.nn.Parameter(torch.randn([total_classes, sz_embed]))
            self.proxies = torch.nn.Parameter(
                torch.randn([total_classes, total_classes]))
            nn.init.kaiming_normal_(self.proxies, mode='fan_out')
            self.proxy_sm = torch.nn.Softmax(dim=1)


    def _init_probs_prior(self, probs, labs, L, U):
        """ Initiallized probabilities from the softmax layer of the CNN """
        n = len(L) + len(U)
        ps = torch.zeros(n, self.m).to(self.device)
        ps[U, :] = probs[U, :]
        # ps[L, labs] = 1.

        if not self.prox:
            ps[L, labs] = 1.
        else:
            # self.proxies = F.softmax(self.proxies, dim=1)
            ps[L, :] = F.softmax(self.proxies, dim=1)[labs, :]
            # ps[L, :] = self.proxies[labs, :]

        # check if probs sum up to 1.
        assert torch.allclose(ps.sum(dim=1), torch.ones(n).cuda())
        return ps

    def _init_probs_prior_only_classes(self, probs, labs, L, U,
                                       classes_to_use):
        """ Different version of the previous version when it considers only classes in the minibatch,
            surprisingly it works worse than the version that considers all classes """
        n = len(L) + len(U)
        ps = torch.zeros(n, self.m).to(self.device)
        ps[U, :] = probs[
            torch.meshgrid(torch.tensor(U), torch.from_numpy(classes_to_use))]
        # ps[L, labs] = 1.

        if not self.prox:
            ps[L, labs] = 1.
        else:
            # self.proxies = F.softmax(self.proxies, dim=1)
            # ps[L, :] = self.proxies[labs, :]
            ps[L, :] = F.softmax(self.proxies, dim=1)[labs, :]

        ps /= ps.sum(dim=ps.dim() - 1).unsqueeze(ps.dim() - 1)
        return ps


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


class TransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, d_embed, num_layers=4, nhead=4, norm=None, num_classes=1367):
        super(TransformerEncoder, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_embed=d_embed, nhead=nhead)
        self.layers = ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

        if self.neck:
            self.bottleneck = nn.BatchNorm1d(d_embed)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.fc = nn.Linear(d_embed, num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.fc.apply(weights_init_classifier)
        else:
            self.fc = nn.Linear(d_embed, num_classes)

    def forward(self, feature_map: Tensor, output_option) -> tuple:

        output = feature_map

        for mod in self.layers:
            output = mod(output)

        if self.norm is not None:
            output = self.norm(output)

        if self.neck:
            feat = self.bottleneck(output)
        else:
            feat = output

        x = self.fc(feat)

        if output_option == 'norm':
            return x, output
        elif output_option == 'plain':
            return x, F.normalize(output, p=2, dim=1)
        elif output_option == 'neck' and self.neck:
            return x, feat
        elif output_option == 'neck' and not self.neck:
            print("Output option neck only avaiable if bottleneck (neck) is "
                  "enabeled - giving back x and fc7")
            return x, output
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_embed, nhead, d_hid=None, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        d_hid = 4 * d_embed if d_hid is None else d_hid
        self.self_attn = MultiheadAttention(d_embed, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_embed, d_hid)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hid, d_embed)

        self.norm1 = nn.LayerNorm(d_embed)
        self.norm2 = nn.LayerNorm(d_embed)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src: Tensor) -> Tensor:

        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class MultiheadAttention(nn.Module):

    __annotations__ = {
        'bias_k': torch._jit_internal.Optional[torch.Tensor],
        'bias_v': torch._jit_internal.Optional[torch.Tensor],
    }

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_nn.Parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        if not self._qkv_same_embed_dim:
            return F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            return F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.q_linear)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.v_linear)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.k_linear)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.num_heads, self.head_dim)
        q = self.q_linear(q).view(bs, -1, self.num_heads, self.head_dim)
        v = self.v_linear(v).view(bs, -1, self.num_heads, self.head_dim)

        # transpose to get dimensions bs * h * sl * embed_dim
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)  # calculate attention using function we will define next
        scores = attention(q, k, v, self.head_dim, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.embed_dim)

        output = self.out(concat)

        return output


def attention(q, k, v, head_dim, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output

class _LinearWithBias(nn.Linear):
    bias: Tensor

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=True)
