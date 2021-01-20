import torch
import torch.nn as nn


class ProxyGen(torch.nn.Module):
    """
        Core Message Passing Network Class. Extracted from torch_geometric, with minor modifications.
        (https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/meta.py)
    """

    def __init__(self, num_classes=None, embedding_dim=None):
        super(ProxyGen, self).__init__()
        self.proxies = torch.nn.Parameter(torch.randn([num_classes, embedding_dim]))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

    def forward(self, labs):
        return torch.cat([self.proxies[i].unsqueeze(0) for i in torch.unique(labs)], dim=0), torch.cat([labs, torch.unique(labs)], dim=0)
