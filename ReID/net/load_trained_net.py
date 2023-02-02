import torch
import os
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
import torch.nn as nn
from .utils import weights_init_kaiming, weights_init_classifier


def load_net(nb_classes, net_type, neck=0, pretrained_path=None, red=1,
             add_distractors=False, pool='avg'):

    # initialize network
    if net_type == 'resnet18':
        model = resnet18(pretrained=True, neck=neck, red=1)
        sz_embed = dim_fc = int(512/red)

    elif net_type == 'resnet34':
        model = resnet34(pretrained=True, neck=neck, red=1)
        sz_embed = dim_fc = int(512/red)
    
    elif net_type == 'resnet50':
        model = resnet50(pretrained=True, neck=neck, red=red,
            add_distractors=add_distractors, pool=pool)
        sz_embed = dim_fc = int(2048/red)           

    elif net_type == 'resnet101':
        model = resnet101(pretrained=True, neck=neck, red=red)
        sz_embed = dim_fc = int(2048/red)

    elif net_type == 'resnet152':
        model = resnet152(pretrained=True, neck=neck, red=red)
        sz_embed = dim_fc = int(2048/red)
    
    # initialize fc layer and bottleneck
    if neck:
        model.bottleneck = nn.BatchNorm1d(dim_fc)
        model.bottleneck.bias.requires_grad_(False)  # no shift
        model.fc = nn.Linear(dim_fc, nb_classes, bias=False)

        model.bottleneck.apply(weights_init_kaiming)
        model.fc.apply(weights_init_classifier)
    else:
        model.fc = nn.Linear(dim_fc, nb_classes)

    # load pretrained net
    if pretrained_path != 'no':
        if not torch.cuda.is_available():
            state_dict = torch.load(
                pretrained_path, map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(pretrained_path)

        model_state_dict = state_dict['model_state_dict']
        optimizer_state_dict = state_dict['optimizer_state_dict']

        model_state_dict = {k: v for k, v in model_state_dict.items()
            if 'fc' not in k.split('.') and 'fc_person' not in k.split('.')}

        model_dict = model.state_dict()
        model_dict.update(model_state_dict)
        model.load_state_dict(model_dict)
    else:
        optimizer_state_dict = None

    return model, sz_embed, optimizer_state_dict
