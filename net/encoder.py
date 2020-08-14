import torch.nn as nn
from torch import Tensor
import copy
from torch.nn.modules.container import ModuleList
import torch.nn.functional as F
import torch
import math


class TransformerEncoder(nn.Module):
    def __init__(self, d_embed, nhead, num_layers, neck, num_classes):
        super(TransformerEncoder, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_embed=d_embed, nhead=nhead)
        self.layers = ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.neck = neck

        if self.neck:
            self.bottleneck = nn.BatchNorm1d(d_embed)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.fc = nn.Linear(d_embed, num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.fc.apply(weights_init_classifier)
        else:
            self.fc = nn.Linear(d_embed, num_classes)

    def forward(self, feature_map: Tensor, output_option) -> tuple:

        for layer in self.layers:
            feature_map = layer(feature_map)

        if self.neck:
            feat = self.bottleneck(feature_map)
        else:
            feat = feature_map

        x = self.fc(feat)

        if output_option == 'norm':
            return x, feature_map
        elif output_option == 'plain':
            return x, F.normalize(feature_map, p=2, dim=1)
        elif output_option == 'neck' and self.neck:
            return x, feat
        elif output_option == 'neck' and not self.neck:
            print("Output option neck only avaiable if bottleneck (neck) is "
                  "enabeled - giving back x and fc7")
            return x, feature_map


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_embed, nhead, d_hid=None, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        d_hid = 4 * d_embed if d_hid is None else d_hid
        self.self_attn = MultiHeadAttention(d_embed, nhead, dropout=dropout)
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


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.constant_(self.q_linear.bias, 0.)

        self.v_linear = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.constant_(self.v_linear.bias, 0.)

        self.k_linear = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.constant_(self.k_linear.bias, 0.)

        self.dropout = nn.Dropout(dropout)

        self.out = nn.Linear(embed_dim, embed_dim)
        nn.init.constant_(self.out.bias, 0.)
    
    def forward(self, q, k, v):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, self.num_heads, self.head_dim)
        q = self.q_linear(q).view(bs, self.num_heads, self.head_dim)
        v = self.v_linear(v).view(bs, self.num_heads, self.head_dim)

        # transpose to get dimensions h * bs * embed_dim
        k = k.transpose(0, 1)
        q = q.transpose(0, 1)
        v = v.transpose(0, 1)  # calculate attention using function we will define next

        scores = attention(q, k, v, self.head_dim, dropout=self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(0, 1).contiguous() \
            .view(bs, self.embed_dim)
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

