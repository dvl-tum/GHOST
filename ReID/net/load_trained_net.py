import torch
import os
#import net
from .embed import embed
from .inception_bn import bn_inception, Inception_embed#, bn_inception_augmented
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .resnet_fpn import ResNetFPN
from .resnet_attention import resnet50 as resnet50_attention
from .densenet import densenet121, densenet161, densenet169, densenet201
from .faster_rcnn import fasterrcnn_resnet50_fpn, FastRCNNPredictor
from .gnn_base import GNNReID
from .graph_generator import GraphGenerator
import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm
from .utils import weights_init_kaiming, weights_init_classifier

def load_net(dataset, nb_classes, mode, attention, net_type, bn_inception={'embed': 0, 'sz_embedding': 512},
             last_stride=0, neck=0, pretrained_path=None, weight_norm=0, final_drop=0.5, stoch_depth=0.8, red=1):
    if net_type == 'bn_inception':
        sz_embed = 1024
        model = net.bn_inception(pretrained=True)
        model.last_linear = nn.Linear(1024, nb_classes)
        
        if bn_inception['embed']:
            model = net.Inception_embed(model, 1024, bn_inception['sz_embedding'], num_classes=nb_classes, neck=neck)
            sz_embed = bn_inception['sz_embedding']

        if pretrained_path != 'no':
            model.load_state_dict(torch.load(pretrained_path))
            
    elif net_type == 'resnet18':
        red = 1
        sz_embed = int(512/red)
        model = net.resnet18(pretrained=True, last_stride=last_stride, neck=neck, final_drop=final_drop, stoch_depth=stoch_depth, red=1)
        dim = int(512/red)
        if neck:
            model.bottleneck = nn.BatchNorm1d(dim)
            model.bottleneck.bias.requires_grad_(False)  # no shift
            if weight_norm:
                model.fc = weightNorm(nn.Linear(dim, nb_classes, bias=False), name = "weight")
            else:
                model.fc = nn.Linear(dim, nb_classes, bias=False)

            model.bottleneck.apply(weights_init_kaiming)
            model.fc.apply(weights_init_classifier)
        else:
            if weight_norm:
                model.fc = weightNorm(nn.Linear(dim, nb_classes), name = "weight")
            else:
                model.fc = nn.Linear(dim, nb_classes)

        if pretrained_path != 'no':
            model.load_state_dict(torch.load(pretrained_path))

    elif net_type == 'resnet34':
        red = 1
        sz_embed = int(512/red)
        model = net.resnet34(pretrained=True, last_stride=last_stride, neck=neck, final_drop=final_drop, stoch_depth=stoch_depth, red=1)
        dim = int(512/red)
        if neck:
            model.bottleneck = nn.BatchNorm1d(dim)
            model.bottleneck.bias.requires_grad_(False)  # no shift
            if weight_norm:
                model.fc = weightNorm(nn.Linear(dim, nb_classes, bias=False), name = "weight")
            else:
                model.fc = nn.Linear(dim, nb_classes, bias=False)

            model.bottleneck.apply(weights_init_kaiming)
            model.fc.apply(weights_init_classifier)
        else:
            if weight_norm:
                model.fc = weightNorm(nn.Linear(dim, nb_classes), name = "weight")
            else:
                model.fc = nn.Linear(dim, nb_classes)

        if pretrained_path != 'no':
            model.load_state_dict(torch.load(pretrained_path))

    elif net_type == 'resnet50':
        sz_embed = int(2048/red)
        model = resnet50(pretrained=True, last_stride=last_stride, neck=neck, final_drop=final_drop, stoch_depth=stoch_depth, red=red, attention=attention)

        dim = int(2048/red)
        print("Dimension of Resnet output {}".format(dim))
        if neck:
            model.bottleneck = nn.BatchNorm1d(dim)
            model.bottleneck.bias.requires_grad_(False)  # no shift
            if weight_norm:
                model.fc = weightNorm(nn.Linear(dim, nb_classes, bias=False), name = "weight")
            else:
                model.fc = nn.Linear(dim, nb_classes, bias=False)

            model.bottleneck.apply(weights_init_kaiming)
            model.fc.apply(weights_init_classifier)
        else: 
            if weight_norm:
                model.fc = weightNorm(nn.Linear(dim, nb_classes), name = "weight")
            else:
                model.fc = nn.Linear(dim, nb_classes)

        if pretrained_path != 'no':
            if not torch.cuda.is_available(): 
                state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
            else:
                state_dict = torch.load(pretrained_path)

            state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k.split('.')}
            #state_dict = {k[7:]: v for k, v in state_dict.items()}
            model_dict = model.state_dict()
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)

    elif net_type == 'resnet50_attention':
        sz_embed = int(2048/red)
        model = resnet50_attention(pretrained=True, last_stride=last_stride, neck=neck, final_drop=final_drop, stoch_depth=stoch_depth, red=red)
        
        dim = int(2048/red)
        if neck:
            model.bottleneck = nn.BatchNorm1d(dim)
            model.bottleneck.bias.requires_grad_(False)  # no shift
            model.fc = nn.Linear(dim, nb_classes, bias=False)

            model.bottleneck.apply(weights_init_kaiming)
            model.fc.apply(weights_init_classifier)
        else: 
            model.fc = nn.Linear(dim, nb_classes)

        if not mode  == 'pretraining' and pretrained_path != 'no':
            no_load = ['fc.bias', 'fc.weight']
            load_dict = {k: v for k, v in torch.load(pretrained_path).items() if k not in no_load}
            for k, v in load_dict.items():
                print(k, v.shape)
            print(model)
            model_dict = model.state_dict()
            model_dict.update(load_dict)
            model.load_state_dict(model_dict)
            #model.load_state_dict(torch.load(pretrained_path))
    elif net_type == 'resnet101':
        sz_embed = int(2048/red)
        model = net.resnet101(pretrained=True, last_stride=last_stride, neck=neck, final_drop=final_drop, stoch_depth=stoch_depth, red=red)

        dim = int(2048/red)
        if neck:
            model.bottleneck = nn.BatchNorm1d(dim)
            model.bottleneck.bias.requires_grad_(False)  # no shift
            if weight_norm:
                model.fc = weightNorm(nn.Linear(dim, nb_classes, bias=False), name = "weight")
            else:
                model.fc = nn.Linear(dim, nb_classes, bias=False)

            model.bottleneck.apply(weights_init_kaiming)
            model.fc.apply(weights_init_classifier)
        else:
            if weight_norm:
                model.fc = weightNorm(nn.Linear(dim, nb_classes), name = "weight")
            else:
                model.fc = nn.Linear(dim, nb_classes)

        if pretrained_path != 'no':
            model.load_state_dict(torch.load(pretrained_path))

    elif net_type == 'resnet152':
        sz_embed = int(2048/red)
        model = net.resnet152(pretrained=True, last_stride=last_stride, neck=neck, final_drop=final_drop, stoch_depth=stoch_depth, red=red)
        dim = int(2048/red)
        if neck:
            model.bottleneck = nn.BatchNorm1d(dim)
            model.bottleneck.bias.requires_grad_(False)  # no shift
            if weight_norm:
                model.fc = weightNorm(nn.Linear(dim, nb_classes, bias=False), name = "weight")
            else:
                model.fc = nn.Linear(dim, nb_classes, bias=False)

            model.bottleneck.apply(weights_init_kaiming)
            model.fc.apply(weights_init_classifier)
        else:
            if weight_norm:
                model.fc = weightNorm(nn.Linear(dim, nb_classes), name = "weight")
            else:
                model.fc = nn.Linear(dim, nb_classes)

        if pretrained_path != 'no':
            model.load_state_dict(torch.load(pretrained_path))

    elif net_type == 'densenet121':
        sz_embed = int(1024/red)
        model = net.densenet121(pretrained=True, last_stride=last_stride, neck=neck, red=red)
        dim = int(1024/red)
        if neck:
            model.bottleneck = nn.BatchNorm1d(dim)
            model.bottleneck.bias.requires_grad_(False)  # no shift
            if weight_norm:
                model.classifier = weightNorm(nn.Linear(dim, nb_classes, bias=False), name = "weight")
            else:
                model.classifier = nn.Linear(dim, nb_classes, bias=False)

            model.bottleneck.apply(weights_init_kaiming)
            model.classifier.apply(weights_init_classifier)
        else:
            if weight_norm:
                model.classifier = weightNorm(nn.Linear(dim, nb_classes), name = "weight")
            else:
                model.classifier = nn.Linear(dim, nb_classes)
        
        if pretrained_path != 'no':
            model.load_state_dict(torch.load(pretrained_path))

    elif net_type == 'densenet161':
        sz_embed = int(2208/red)
        model = net.densenet161(pretrained=True, last_stride=last_stride, neck=neck, red=red)
        dim = int(2208/red)
        if neck:
            model.bottleneck = nn.BatchNorm1d(dim)
            model.bottleneck.bias.requires_grad_(False)  # no shift
            if weight_norm:
                model.classifier = weightNorm(nn.Linear(dim, nb_classes, bias=False), name = "weight")
            else:
                model.classifier = nn.Linear(dim, nb_classes, bias=False)

            model.bottleneck.apply(weights_init_kaiming)
            model.classifier.apply(weights_init_classifier)
        else:
            if weight_norm:
                model.classifier = weightNorm(nn.Linear(dim, nb_classes), name = "weight")
            else:
                model.classifier = nn.Linear(dim, nb_classes)

        if pretrained_path != 'no':
            model.load_state_dict(torch.load(pretrained_path))

    elif net_type == 'densenet169':
        sz_embed = int(1664/red)
        model = net.densenet169(pretrained=True, last_stride=last_stride, neck=neck, red=red)
        dim = int(1664/red)
        if neck:
            model.bottleneck = nn.BatchNorm1d(dim)
            model.bottleneck.bias.requires_grad_(False)  # no shift
            if weight_norm:
                model.classifier = weightNorm(nn.Linear(dim, nb_classes, bias=False), name = "weight")
            else:
                model.classifier = nn.Linear(dim, nb_classes, bias=False)

            model.bottleneck.apply(weights_init_kaiming)
            model.classifier.apply(weights_init_classifier)
        else:
            if weight_norm:
                model.classifier = weightNorm(nn.Linear(dim, nb_classes), name = "weight")
            else:
                model.classifier = nn.Linear(dim, nb_classes)

        if pretrained_path != 'no':
            model.load_state_dict(torch.load(pretrained_path))

    elif net_type == 'densenet201':
        sz_embed = int(1920)
        model = net.densenet201(pretrained=True, last_stride=last_stride, neck=neck, red=red)
        dim = int(1920/red)
        if neck:
            model.bottleneck = nn.BatchNorm1d(dim)
            model.bottleneck.bias.requires_grad_(False)  # no shift
            if weight_norm:
                model.classifier = weightNorm(nn.Linear(dim, nb_classes, bias=False), name = "weight")
            else:
                model.classifier = nn.Linear(dim, nb_classes, bias=False)

            model.bottleneck.apply(weights_init_kaiming)
            model.classifier.apply(weights_init_classifier)
        else:
            if weight_norm:
                model.classifier = weightNorm(nn.Linear(dim, nb_classes), name = "weight")
            else:
                model.classifier = nn.Linear(dim, nb_classes)

        if pretrained_path != 'no':
            model.load_state_dict(torch.load(pretrained_path))

    elif net_type == 'resnet50FPN':
        sz_embed = int(256/red)
        model = ResNetFPN(pretrained=True, neck=neck, red=red, attention=attention)

        dim = int(256/red)
        if model.combine or model.conv_combi or model.last_only:
            if model.combi_big and model.last_only:
                model.last_linear = nn.Linear(int(256/red), 2048)
            if model.make_small and model.last_only:
                model.last_linear = nn.Linear(int(256/red), 128)
            if model.neck:
                if model.combi_big:
                    bottleneck = nn.BatchNorm1d(2048)
                    bottleneck.bias.requires_grad_(False)  # no shift
                    fc = nn.Linear(2048, nb_classes,
                                        bias=False)
                elif model.last_only:
                    bottleneck = nn.BatchNorm1d(256)
                    bottleneck.bias.requires_grad_(False)  # no shift
                    fc = nn.Linear(256, nb_classes,
                                        bias=False)
                else:
                    bottleneck = nn.BatchNorm1d(4 * dim)
                    bottleneck.bias.requires_grad_(False)  # no shift
                    fc = nn.Linear(4 * dim, nb_classes,
                                        bias=False)
                bottleneck.apply(weights_init_kaiming)
                fc.apply(weights_init_classifier)
                model.necks = bottleneck
                model.fcs = fc
            else:
                if model.combi_big:
                    fc = nn.Linear(2048, nb_classes)
                elif model.last_only:
                    if model.make_small:
                        fc = nn.Linear(128, nb_classes)
                    else:
                        fc = nn.Linear(256, nb_classes)
                else:
                    fc = nn.Linear(4 * dim, nb_classes)
                model.fcs = fc
        else:
            fcs = []
            neck = []
            for _ in range(4):
                if model.neck:
                    bottleneck = nn.BatchNorm1d(dim)
                    bottleneck.bias.requires_grad_(False)  # no shift
                    fc = nn.Linear(dim, nb_classes,
                                        bias=False)

                    bottleneck.apply(weights_init_kaiming)
                    fc.apply(weights_init_classifier)
                    neck.append(bottleneck)
                else:
                    fc = nn.Linear(dim, nb_classes)
                fcs.append(fc)
            if model.neck:
                model.necks = nn.Sequential(*neck)
            model.fcs = nn.Sequential(*fcs)
        
        if not mode  == 'pretraining' and pretrained_path != 'no':
            print("loading from {}".format(pretrained_path))
            no_load = [k for k in torch.load(pretrained_path).keys() if 'fc' in k]
            load_dict = {k: v for k, v in torch.load(pretrained_path).items() if k not in no_load}

            model_dict = model.state_dict()
            model_dict.update(load_dict)
            model.load_state_dict(model_dict)

    elif net_type == 'FRCNN_FPN':
        model = fasterrcnn_resnet50_fpn(pretrained=True, num_ids=nb_classes, additional_embedding=True)
        sz_embed = model.roi_heads.box_predictor.embedding_head.out_features

        # training
        # output = model(images, targets)
        # For inference
        # model.eval()
        
    return model, sz_embed



if __name__ == '__main__':
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 2  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)