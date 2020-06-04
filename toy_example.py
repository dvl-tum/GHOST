from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from combine_sampler import CombineSampler
from collections import defaultdict
import gtg as gtg_module
from RAdam import RAdam
import utils
import data_utility
import net



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64, 10)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        fc7 = torch.flatten(x, 1)
        x = self.fc1(fc7)
        # output = F.log_softmax(x, dim=1)
        return x, fc7


def train(model, device, train_loader, opt, gtg, center, lab_smooth, nb_classes, num_lab, temp):
    criterion = nn.NLLLoss()

    # add label smoothing
    if lab_smooth:
        criterion2 = utils.CrossEntropyLabelSmooth(
            num_classes=nb_classes)
    else:
        criterion2 = nn.CrossEntropyLoss()

    # add center loss
    if center:
        criterion3 = utils.CenterLoss(num_classes=nb_classes)

    model.train()
    activations = list()
    for x, Y in train_loader:
        Y = Y.to(device)
        opt.zero_grad()

        probs, fc7 = model(x.to(device))
        activations.append([[fc7[i, 0], fc7[i, 1]] for i in range(fc7.shape[0])])
        loss = criterion2(probs, Y)

        labs, L, U = data_utility.get_labeled_and_unlabeled_points(
            labels=Y,
            num_points_per_class=num_lab,
            num_classes=nb_classes)

        # compute the smoothed softmax
        probs_for_gtg = F.softmax(probs / temp)

        # do GTG (iterative process)
        probs_for_gtg, W = gtg(fc7, fc7.shape[0], labs, L, U,
                               probs_for_gtg)
        probs_for_gtg = torch.log(probs_for_gtg + 1e-12)

        # compute the losses
        loss1 = criterion(probs_for_gtg, Y)
        loss = loss1 + loss
        # add center loss
        if center:
            loss += criterion3(fc7, Y)

        else:
            loss.backward()
        opt.step()

    return activations

def main():
    # Training settings
    nb_classes = 10
    lab_smooth = 0
    center = 0

    config = {'lr': 4.4819286767613e-05,
              'weight_decay': 1.5288509425482333e-13,  # rest does not matter
              'num_classes_iter': 5,
              'num_elements_class': 5,
              'num_labeled_points_class': 2,
              'num_iter_gtg': 1,
              'temperature': 80}

    use_cuda = torch.cuda.is_available()

    torch.manual_seed(1)

    device = torch.device("cuda" if use_cuda else "cpu")
    dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    dataset_test = datasets.MNIST('../data', train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]))

    ddict = defaultdict(list)
    for idx, label in enumerate(dataset.train_labels):
        ddict[label.data.item()].append(idx)

    list_of_indices_for_each_class = []
    for key in ddict:
        list_of_indices_for_each_class.append(ddict[key])

    sampler = CombineSampler(list_of_indices_for_each_class,
                             config['num_classes_iter'],
                             config['num_elements_class'])

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(dataset,
                                               sampler=sampler,
                                               batch_size=5*5,
                                               shuffle=False,
                                               drop_last=True,
                                               **kwargs)

    model = Net().to(device)

    '''model = net.load_net(dataset=None,
                         net_type='resnet50',
                         nb_classes=nb_classes,
                         last_stride=0,
                         neck=0,
                         use_pretrained=0,
                         bn_GL=0)
    model = model.to(device)'''

    gtg = gtg_module.GTG(nb_classes,
                         max_iter=config['num_iter_gtg'],
                         sim='correlation',
                         set_negative='hard',
                         device=device)
    opt = RAdam(
        [{'params': list(set(model.parameters())), 'lr': config['lr']}],
        weight_decay=config['weight_decay'])

    for epoch in range(1, 2 + 1):
        print(epoch)
        activations = train(model, device, train_loader, opt, gtg, center, lab_smooth, nb_classes, config['num_labeled_points_class'], config['temperature'])

    x = [a[0] for a in activations]
    y = [a[1] for a in activations]

    import matplotlib.pyplot as plt
    plt.plot(x, y)
    plt.show()

if __name__ == '__main__':
    main()