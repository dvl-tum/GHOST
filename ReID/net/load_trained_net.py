import torch
import os
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
import torch.nn as nn
from .utils import weights_init_kaiming, weights_init_classifier


def load_net(nb_classes, net_type, neck=0, pretrained_path=None, red=1,
             add_distractors=False, pool='avg'):

    if net_type == 'resnet18':
        red = 1
        sz_embed = int(512/red)
        model = resnet18(pretrained=True, neck=neck, red=1)
        dim = int(512/red)
        if neck:
            model.bottleneck = nn.BatchNorm1d(dim)
            model.bottleneck.bias.requires_grad_(False)  # no shift
            model.fc = nn.Linear(dim, nb_classes, bias=False)

            model.bottleneck.apply(weights_init_kaiming)
            model.fc.apply(weights_init_classifier)
        else:
            model.fc = nn.Linear(dim, nb_classes)

        if pretrained_path != 'no':
            model.load_state_dict(torch.load(pretrained_path))

    elif net_type == 'resnet34':
        red = 1
        sz_embed = int(512/red)
        model = resnet34(pretrained=True, neck=neck, red=1)
        dim = int(512/red)
        if neck:
            model.bottleneck = nn.BatchNorm1d(dim)
            model.bottleneck.bias.requires_grad_(False)  # no shift
            model.fc = nn.Linear(dim, nb_classes, bias=False)

            model.bottleneck.apply(weights_init_kaiming)
            model.fc.apply(weights_init_classifier)
        else:
            model.fc = nn.Linear(dim, nb_classes)

        if pretrained_path != 'no':
            model.load_state_dict(torch.load(pretrained_path))

    elif net_type == 'resnet50':
        model = resnet50(pretrained=True, neck=neck, red=red,
            add_distractors=add_distractors, pool=pool)

        sz_embed = dim_fc = int(2048/red)
        print("Dimension of Resnet output {}".format(dim_fc))
        if neck:
            model.bottleneck = nn.BatchNorm1d(dim_fc)
            model.bottleneck.bias.requires_grad_(False)  # no shift
            model.fc = nn.Linear(dim_fc, nb_classes, bias=False)

            model.bottleneck.apply(weights_init_kaiming)
            model.fc.apply(weights_init_classifier)
        else:
            model.fc = nn.Linear(dim_fc, nb_classes)

        if pretrained_path != 'no':
            if not torch.cuda.is_available():
                state_dict = torch.load(
                    pretrained_path, map_location=torch.device('cpu'))
            else:
                state_dict = torch.load(pretrained_path)

            state_dict = {k: v for k, v in state_dict.items()
                if 'fc' not in k.split('.') and 'fc_person' not in k.split('.')}

            model_dict = model.state_dict()
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)

    elif net_type == 'resnet101':
        sz_embed = int(2048/red)
        model = resnet101(pretrained=True, neck=neck, red=red)

        dim = int(2048/red)
        if neck:
            model.bottleneck = nn.BatchNorm1d(dim)
            model.bottleneck.bias.requires_grad_(False)  # no shift
            model.fc = nn.Linear(dim, nb_classes, bias=False)

            model.bottleneck.apply(weights_init_kaiming)
            model.fc.apply(weights_init_classifier)
        else:
            model.fc = nn.Linear(dim, nb_classes)

        if pretrained_path != 'no':
            model.load_state_dict(torch.load(pretrained_path))

    elif net_type == 'resnet152':
        sz_embed = int(2048/red)
        model = resnet152(pretrained=True, neck=neck, red=red)
        dim = int(2048/red)
        if neck:
            model.bottleneck = nn.BatchNorm1d(dim)
            model.bottleneck.bias.requires_grad_(False)  # no shift
            model.fc = nn.Linear(dim, nb_classes, bias=False)

            model.bottleneck.apply(weights_init_kaiming)
            model.fc.apply(weights_init_classifier)
        else:
            model.fc = nn.Linear(dim, nb_classes)

        if pretrained_path != 'no':
            model.load_state_dict(torch.load(pretrained_path))

    return model, sz_embed
