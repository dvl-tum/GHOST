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
    def __init__(self, total_classes, tol=-1., max_iter=5, sim='correlation', set_negative='hard', mode='replicator', device='cuda:0'):
        super(GTG, self).__init__()
        self.m = total_classes
        self.tol = tol
        self.max_iter = max_iter
        self.mode = mode
        self.sim = sim
        self.set_negative = set_negative
        self.device = device

    def _init_probs(self, labs, L, U):
        n = len(L) + len(U)
        ps = torch.zeros(n, self.m).to(self.device)
        ps[U, :] = 1. / self.m
        ps[L, labs] = 1.

        # check if probs sum up to 1.
        assert torch.allclose(ps.sum(dim=1), torch.ones(n))
        return ps

    def _init_probs_prior(self, probs, labs, L, U):
        n = len(L) + len(U)
        ps = torch.zeros(n, self.m).to(self.device)
        ps[U, :] = probs[U, :]
        ps[L, labs] = 1.

        # check if probs sum up to 1.
        assert torch.allclose(ps.sum(dim=1), torch.ones(n).cuda())
        return ps

    def _init_probs_prior_only_classes(self, probs, labs, L, U, classes_to_use):
        n = len(L) + len(U)
        ps = torch.zeros(n, self.m).to(self.device)
        ps[U, :] = probs[torch.meshgrid(torch.tensor(U), torch.from_numpy(classes_to_use))]
        ps[L, labs] = 1.
        ps /= ps.sum(dim=ps.dim() - 1).unsqueeze(ps.dim() - 1)
        return ps

    def set_negative_to_zero_old(self, W):
        n = W.shape[0]
        mask = torch.zeros((n, n), requires_grad=False).cuda()
        for i in range(n):
            for j in range(i + 1, n):
                if W[i, j] > 0:
                    mask[i, j] = mask[j, i] = 1.
        return W * mask

    def set_negative_to_zero(self, W):
        return F.relu(W)

    def set_negative_to_zero_soft(self, W):
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
        else:
            n = x.shape[0]
            W = torch.zeros(n, n)
            for i, xi in enumerate(x):
                for j, xj in enumerate(x[(i + 1):], i + 1):
                    if self.sim == 'learnt':
                        W[i, j] = W[j, i] = self.sim(xi, xj) + 1e-8
                    elif self.sim == 'icc':
                        W[i, j] = W[j, i] = self.compute_icc(xi, xj)
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
