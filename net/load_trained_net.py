import torch
import os
import net
import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm


def load_net(dataset, nb_classes, mode, net_type, bn_inception={'embed': 0, 'sz_embedding': 512},
             last_stride=0, neck=0, pretrained_path=None, weight_norm=0):
    if net_type == 'bn_inception':
        sz_embed = 1024
        model = net.bn_inception(pretrained=True)
        model.last_linear = nn.Linear(1024, nb_classes)
        if not mode  == 'pretraining':
            model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))

        if bn_inception['embed']:
            model = net.Inception_embed(model, 1024, bn_inception['sz_embedding'], num_classes=nb_classes)
            if not mode  == 'pretraining':
                model.load_state_dict(torch.load(os.path.join('net', 'finetuned_cub_embedded_512_10_.pth')))

    elif net_type == 'resnet18':
        sz_embed = 512
        model = net.resnet18(pretrained=True)
        model.fc = nn.Linear(512, nb_classes)
        if not mode  == 'pretraining':
            model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))

    elif net_type == 'resnet34':
        sz_embed = 512
        model = net.resnet34(pretrained=True)
        model.fc = nn.Linear(512, nb_classes)
        if not mode  == 'pretraining':
            model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))

    elif net_type == 'resnet50':
        sz_embed = 2048
        model = net.resnet50(pretrained=True, last_stride=last_stride, neck=neck)
        #model.fc = nn.Linear(2048, nb_classes)
        #if neck: model.bottleneck = nn.BatchNorm1d(2048)
        if neck:
            model.bottleneck = nn.BatchNorm1d(2048)
            model.bottleneck.bias.requires_grad_(False)  # no shift
            if weight_norm:
                model.fc = weightNorm(nn.Linear(2048, nb_classes, bias=False), name = "weight")
            else:
                model.fc = nn.Linear(2048, nb_classes, bias=False)

            model.bottleneck.apply(weights_init_kaiming)
            model.fc.apply(weights_init_classifier)
        else: 
            if weight_norm:
                model.fc = weightNorm(nn.Linear(2048, nb_classes), name = "weight")
            else:
                model.fc = nn.Linear(2048, nb_classes)

        if not mode  == 'pretraining' and pretrained_path != 'no':
            no_load = ["linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias",
                        "bottleneck.weight", "bottleneck.bias", "bottleneck.running_mean",
                        "bottleneck.running_var", "bottleneck.num_batches_tracked", "fc.weight"]
            no_load = ['fc.bias']
            no_load = []
            load_dict = {k: v for k, v in torch.load(pretrained_path).items() if k not in no_load}
            model_dict = model.state_dict()
            model_dict.update(load_dict)
            model.load_state_dict(model_dict)
            #model.load_state_dict(torch.load(pretrained_path))

    elif net_type == 'resnet101':
        sz_embed = 2048
        model = net.resnet101(pretrained=True)
        model.fc = nn.Linear(2048, nb_classes)
        if not mode  == 'pretraining':
            model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))

    elif net_type == 'resnet152':
        sz_embed = 2048
        model = net.resnet152(pretrained=True)
        model.fc = nn.Linear(2048, nb_classes)
        if not mode  == 'pretraining':
            model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))

    elif net_type == 'densenet121':
        sz_embed = 1024
        model = net.densenet121(pretrained=True, last_stride=last_stride, neck=neck)

        if neck:
            model.bottleneck = nn.BatchNorm1d(1024)
            model.bottleneck.bias.requires_grad_(False)  # no shift
            model.classifier = nn.Linear(1024, nb_classes, bias=False)

            model.bottleneck.apply(weights_init_kaiming)
            model.classifier.apply(weights_init_classifier)
        else: model.classifier = nn.Linear(1024, nb_classes)

        if not mode  == 'pretraining' and pretrained_path != 'no':
            model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))
        # model.load_state_dict(torch.load('sop_resnet_new/Stanford_paramRes_16bit_densenet121_0.0002_5.2724883734490575e-12_10_6_2.pth'))

    elif net_type == 'densenet161':
        sz_embed = 2208
        model = net.densenet161(pretrained=True, last_stride=last_stride, neck=neck)
        if neck:
            model.bottleneck = nn.BatchNorm1d(2208)
            model.bottleneck.bias.requires_grad_(False)  # no shift
            model.classifier = nn.Linear(2208, nb_classes, bias=False)

            model.bottleneck.apply(weights_init_kaiming)
            model.classifier.apply(weights_init_classifier)
        else: model.classifier = nn.Linear(2208, nb_classes)
        if not mode  == 'pretraining' and pretrained_path != 'no':
            print(pretrained_path)
            model.load_state_dict(torch.load(pretrained_path))
        # model.load_state_dict(torch.load('sop_resnet_new/Stanford_paramRes_16bit_densenet161_0.0002_5.2724883734490575e-12_10_6_2.pth'))

    elif net_type == 'densenet169':
        sz_embed = 1664
        model = net.densenet169(pretrained=True, last_stride=last_stride, neck=neck)
        if neck:
            model.bottleneck = nn.BatchNorm1d(1664)
            model.bottleneck.bias.requires_grad_(False)  # no shift
            model.classifier = nn.Linear(1664, nb_classes, bias=False)

            model.bottleneck.apply(weights_init_kaiming)
            model.classifier.apply(weights_init_classifier)
        else: model.classifier = nn.Linear(1664, nb_classes)
        if not mode  == 'pretraining' and pretrained_path != 'no':
            model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))
        # model.load_state_dict(torch.load('sop_resnet_new/Stanford_paramRes_16bit_densenet169_0.0002_5.2724883734490575e-12_10_6_2.pth'))

    elif net_type == 'densenet201':
        sz_embed = 1920
        model = net.densenet201(pretrained=True, last_stride=last_stride, neck=neck)
        if neck:
            model.bottleneck = nn.BatchNorm1d(1920)
            model.bottleneck.bias.requires_grad_(False)  # no shift
            model.classifier = nn.Linear(1920, nb_classes, bias=False)

            model.bottleneck.apply(weights_init_kaiming)
            model.classifier.apply(weights_init_classifier)
        else: model.classifier = nn.Linear(1920, nb_classes)
        if not mode  == 'pretraining' and pretrained_path != 'no':
            model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))
        # model.load_state_dict(torch.load('sop_resnet_new/Stanford_paramRes_16bit_densenet201_0.0002_5.2724883734490575e-12_10_6_2.pth'))

    return model, sz_embed


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
