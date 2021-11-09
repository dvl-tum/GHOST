import torch
import torch.nn.functional as F


class GraphGenerator():
    def __init__(self, dev, thresh=0, sim_type='correlation', set_negative='hard'):
        self.dev = dev
        self.thresh = thresh
        self.sim = sim_type
        self.set_negative  = set_negative

    @staticmethod
    def set_negative_to_zero(W):
        return F.relu(W)

    def set_negative_to_zero_soft(self, W):
        """ It shifts the negative probabilities towards the positive regime """
        n = W.shape[0]
        minimum = torch.min(W)
        W = W - minimum
        W = W * (torch.ones((n, n)).to(self.device) - torch.eye(n).to(self.device))
        return W

    def _get_A(self, W):
        W = torch.where(W > self.thresh, W, torch.tensor(0).float().to(self.dev))
        A = torch.ones_like(W).where(W > self.thresh, torch.tensor(0).float().to(self.dev))
        
        return W, A

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
            W = self.set_negative_to_zero(W.to(self.dev))
        else:
            W = self.set_negative_to_zero_soft(W)

        return W

    def get_graph(self, x, Y=None, num_dets=None):
        #W = self._get_W(x)
        #W, A = self._get_A(W)
        #A = torch.ones((x.shape[0], x.shape[0])).to(x.get_device())
        A = torch.eye(x.shape[0]).to(x.get_device())
        W = A
        n = W.shape[0]
        
        A += 0.000001
        A = torch.nonzero(A)
        if num_dets is not None:
            A_self = torch.eye(n)
            A_self = torch.nonzero(A_self).to(self.dev)
            W_self = W[A_self[:, 0], A_self[:, 1]].to(self.dev)

        W = W[A[:, 0], A[:, 1]]
        if num_dets is not None:
            W = W[A[:, 0] >= num_dets]
            A = A[A[:, 0] >= num_dets]
            W = W[A[:, 1] < num_dets]
            A = A[A[:, 1] < num_dets]
            A = torch.cat([A, A_self], dim=0)
            W = torch.cat([W, W_self], dim=0)

        return W, A, x
