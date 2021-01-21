from torch import nn
import torch
from torch_scatter import scatter_max, scatter_add
import numpy as np
import logging
import math

logger = logging.getLogger('GNNReID.Util')


class Sequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class SequentialInter(nn.Sequential):
    def forward(self, *inputs):
        out = list()
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
                out.append(inputs[0])
            else:
                inputs = module(inputs)
                out.append(inputs[0])
        print(inputs)
        quit()
        return inputs


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
                    torch.sqrt(x.var(unbiased=False, dim=-1, keepdim=True) + self.eps))

        x = x.view(init_shape)

        if self.affine:
            x *= self.weight
            x += self.bias

        return x

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)


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


def softmax(src, index, dim, dim_size, margin: float = 0.):
    src_max = torch.clamp(scatter_max(src.float(), index, dim=dim, dim_size=dim_size)[0], min=0.)
    src = (src - src_max.index_select(dim=dim, index=index)).exp()
    denom = scatter_add(src, index, dim=dim, dim_size=dim_size)
    out = src / (denom + (margin - src_max).exp()).index_select(dim, index)

    return out
