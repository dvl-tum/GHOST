import torch.nn as nn
import torch
import dynamics
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F


class NonLinearSimilarity(nn.Module):
    def __init__(self, in_features):
        super(NonLinearSimilarity, self).__init__()
        # self.bn = nn.BatchNorm1d(in_features)
        self.lin = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xi, xj):
        out = torch.exp(xi - xj)
        out = self.lin(out)
        out = self.sigmoid(out)
        return out


class GTG(nn.Module):
    def __init__(self, total_classes, tol=-1., max_iter=5, sim='correlation', set_negative='hard', mode='replicator', device='cuda:0', sz_embed=2048, proxies=0):
        super(GTG, self).__init__()
        self.m = total_classes
        self.tol = tol
        self.max_iter = max_iter
        self.mode = mode
        self.sim = sim
        self.set_negative = set_negative
        self.device = device
        self.prox = proxies
        if self.prox:
            self.proxies = torch.nn.Parameter(torch.randn(total_classes, sz_embed).cuda())
            nn.init.kaiming_normal_(self.proxies, mode='fan_out')

    def _init_probs_uniform(self, labs, L, U):
        """ Initialized the probabilities of GTG from uniform distribution """
        n = len(L) + len(U)
        ps = torch.zeros(n, self.m).to(self.device)
        ps[U, :] = 1. / self.m
        ps[L, labs] = 1.

        # check if probs sum up to 1.
        assert torch.allclose(ps.sum(dim=1), torch.ones(n))
        return ps

    def _init_probs_prior(self, probs, labs, L, U):
        """ Initiallized probabilities from the softmax layer of the CNN """
        n = len(L) + len(U)
        ps = torch.zeros(n, self.m).to(self.device)
        ps[U, :] = probs[U, :]
        if not self.prox:
            ps[L, labs] = 1.
        else:
            ps[L, :] = self.proxies[labs, :]

        # check if probs sum up to 1.
        assert torch.allclose(ps.sum(dim=1), torch.ones(n).cuda())
        return ps

    def _init_probs_prior_only_classes(self, probs, labs, L, U, classes_to_use):
        """ Different version of the previous version when it considers only classes in the minibatch,
            surprisingly it works worse than the version that considers all classes """
        n = len(L) + len(U)
        ps = torch.zeros(n, self.m).to(self.device)
        ps[U, :] = probs[torch.meshgrid(torch.tensor(U), torch.from_numpy(classes_to_use))]
        if not self.prox:
            ps[L, labs] = 1.
        else:
            ps[L, :] = self.proxies[labs, :]
        ps /= ps.sum(dim=ps.dim() - 1).unsqueeze(ps.dim() - 1)
        return ps

    def set_negative_to_zero(self, W):
        return F.relu(W)

    def set_negative_to_zero_soft(self, W):
        """ It shifts the negative probabilities towards the positive regime """
        n = W.shape[0]
        minimum = torch.min(W)
        W = W - minimum
        W = W * (torch.ones((n, n)).to(self.device) - torch.eye(n).to(self.device))
        return W

    def _get_W(self, x):

        if self.sim == 'correlation':
            x = (x - x.mean(dim=1).unsqueeze(1))
            norms = x.norm(dim=1)
            W = torch.mm(x, x.t()) / torch.ger(norms, norms)
        elif self.sim == 'cosine':
            W = torch.mm(x, x.t())
        elif self.sim == 'learnt':
            n = x.shape[0]
            W = torch.zeros(n, n)
            for i, xi in enumerate(x):
                for j, xj in enumerate(x[(i + 1):], i + 1):
                    W[i, j] = W[j, i] = self.sim(xi, xj) + 1e-8
            W = W.cuda()

        if self.set_negative == 'hard':
            W = self.set_negative_to_zero(W.cuda())
        else:
            W = self.set_negative_to_zero_soft(W)
        return W

    def forward(self, fc7, num_points, labs, L, U, probs=None, classes_to_use=None):
        W = self._get_W(fc7)
        if type(probs) is type(None):
            ps = self._init_probs(labs, L, U).cuda()
        else:
            if type(classes_to_use) is type(None):
                ps = probs
                ps = self._init_probs_prior(ps, labs, L, U)
            else:
                ps = probs
                ps = self._init_probs_prior_only_classes(ps, labs, L, U, classes_to_use)
        ps = dynamics.dynamics(W, ps, self.tol, self.max_iter, self.mode)
        return ps, W
