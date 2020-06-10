import evaluation
import torch
import net
import data_utility
from torch import nn
from torch.autograd import Variable


def predict_batchwise_reid(model, dataloader, test_option='norm'):
    fc7s, L = [], []
    features = dict()
    labels = dict()
    with torch.no_grad():
        for X, Y, P in dataloader:
            if torch.cuda.is_available(): X = X.cuda()
            _, fc7 = model(X, test_option=test_option)
            for path, out, y in zip(P, fc7, Y):
                features[path] = out
                labels[path] = y
            fc7s.append(fc7.cpu())
            L.append(Y)
    fc7, Y = torch.cat(fc7s), torch.cat(L)
    return torch.squeeze(fc7), torch.squeeze(Y), features, labels


def evaluate_reid(model, dataloader, query=None, gallery=None, root=None,
                  test_option='norm'):
    model_is_training = model.training
    model.eval()
    _, _, features, _ = predict_batchwise_reid(model, dataloader, test_option)
    mAP, cmc = evaluation.calc_mean_average_precision(features, query,
                                                      gallery, root)
    model.train(model_is_training)
    return mAP, cmc

# cross entropy and center loss
class CrossEntropyLabelSmooth(torch.nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class CenterLoss(torch.nn.Module):
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = torch.nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = torch.nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        #dist = []
        #for i in range(batch_size):
        #    value = distmat[i][mask[i]]
        #    value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
        #    dist.append(value)
        #dist = torch.cat(dist)
        #loss = dist.mean()
        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        # Compute pairwise distance
        m, n = inputs.size(0), inputs.size(0)
        x = inputs.view(m, -1)
        y = inputs.view(n, -1)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()

        dist.addmm_(1, -2, x, y.t())

        mask = targets.type(torch.ByteTensor).cuda()

        # for numerical stability
        # For each anchor, find the hardest positive and negative
        #mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            mask = targets == targets[i]
            dist_ap.append(dist[i][mask].max()) #hp
            dist_an.append(dist[i][mask == 0].min()) #hn
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()

        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec


if __name__ == '__main__':
    # test create loader and load net
    
    model = net.load_net(dataset='cuhk03',
                         net_type='resnet50',
                         nb_classes=751,
                         embed=False,
                         sz_embedding=512,
                         pretraining=True)

    dl_tr, dl_ev, q, g = data_utility.create_loaders(
        data_root='../../datasets/cuhk03-np/detected',
        input_size=224, size_batch=4, pretraining=False,
        num_workers=2, num_classes_iter=2, num_elements_class=2
        )

    evaluate_reid(model, dl_ev, q, g, '../../datasets/cuhk03-np/detected')
