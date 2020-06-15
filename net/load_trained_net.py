import torch
import os
import net
import torch.nn as nn


def load_net(dataset, net_type, nb_classes, embed=False, sz_embedding=512, pretraining=False, last_stride=0, neck=0, load_path=None, bn_GL=0, use_pretrained=0):

    if net_type == 'bn_inception':
        model = net.bn_inception(pretrained=True)
        model.last_linear = nn.Linear(1024, nb_classes)
        if not pretraining:
            model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))

        if embed:
            model = net.Inception_embed(model, 1024, sz_embedding, num_classes=nb_classes)
            if not pretraining:
                model.load_state_dict(torch.load(os.path.join('net', 'finetuned_cub_embedded_512_10_.pth')))

    elif net_type == 'resnet18':
        model = net.resnet18(pretrained=True)
        model.fc = nn.Linear(512, nb_classes)
        if not pretraining:
            model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))

    elif net_type == 'resnet34':
        model = net.resnet34(pretrained=True)
        model.fc = nn.Linear(512, nb_classes)
        if not pretraining:
            model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))

    elif net_type == 'resnet50':
        model = net.resnet50(pretrained=True, last_stride=last_stride, neck=neck, bn_GL=bn_GL)
        #model.fc = nn.Linear(2048, nb_classes)
        #if neck: model.bottleneck = nn.BatchNorm1d(2048)
        if neck:
            model.bottleneck = nn.BatchNorm1d(2048)
            model.bottleneck.bias.requires_grad_(False)  # no shift
            model.fc = nn.Linear(2048, nb_classes, bias=False)

            model.bottleneck.apply(weights_init_kaiming)
            model.fc.apply(weights_init_classifier)
        else: model.fc = nn.Linear(2048, nb_classes)

        if not pretraining and use_pretrained:
            model.load_state_dict(torch.load(load_path))

    elif net_type == 'resnet101':
        model = net.resnet101(pretrained=True)
        model.fc = nn.Linear(2048, nb_classes)
        if not pretraining:
            model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))

    elif net_type == 'resnet152':
        model = net.resnet152(pretrained=True)
        model.fc = nn.Linear(2048, nb_classes)
        if not pretraining:
            model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))

    elif net_type == 'densenet121':
        model = net.densenet121(pretrained=True, last_stride=last_stride, neck=neck, bn_GL=bn_GL)

        if neck:
            model.bottleneck = nn.BatchNorm1d(1024)
            model.bottleneck.bias.requires_grad_(False)  # no shift
            model.classifier = nn.Linear(1024, nb_classes, bias=False)

            model.bottleneck.apply(weights_init_kaiming)
            model.classifier.apply(weights_init_classifier)
        else: model.classifier = nn.Linear(1024, nb_classes)

        if not pretraining and use_pretrained:
            model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))
        # model.load_state_dict(torch.load('sop_resnet_new/Stanford_paramRes_16bit_densenet121_0.0002_5.2724883734490575e-12_10_6_2.pth'))

    elif net_type == 'densenet161':
        model = net.densenet161(pretrained=True, last_stride=last_stride, neck=neck, bn_GL=bn_GL)
        if neck:
            model.bottleneck = nn.BatchNorm1d(2208)
            model.bottleneck.bias.requires_grad_(False)  # no shift
            model.classifier = nn.Linear(2208, nb_classes, bias=False)

            model.bottleneck.apply(weights_init_kaiming)
            model.classifier.apply(weights_init_classifier)
        else: model.classifier = nn.Linear(2208, nb_classes)
        if not pretraining and use_pretrained:
            model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))
        # model.load_state_dict(torch.load('sop_resnet_new/Stanford_paramRes_16bit_densenet161_0.0002_5.2724883734490575e-12_10_6_2.pth'))

    elif net_type == 'densenet169':
        model = net.densenet169(pretrained=True, last_stride=last_stride, neck=neck, bn_GL=bn_GL)
        if neck:
            model.bottleneck = nn.BatchNorm1d(1664)
            model.bottleneck.bias.requires_grad_(False)  # no shift
            model.classifier = nn.Linear(1664, nb_classes, bias=False)

            model.bottleneck.apply(weights_init_kaiming)
            model.classifier.apply(weights_init_classifier)
        else: model.classifier = nn.Linear(1664, nb_classes)
        if not pretraining and use_pretrained:
            model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))
        # model.load_state_dict(torch.load('sop_resnet_new/Stanford_paramRes_16bit_densenet169_0.0002_5.2724883734490575e-12_10_6_2.pth'))

    elif net_type == 'densenet201':
        model = net.densenet201(pretrained=True, last_stride=last_stride, neck=neck, bn_GL=bn_GL)
        if neck:
            model.bottleneck = nn.BatchNorm1d(1920)
            model.bottleneck.bias.requires_grad_(False)  # no shift
            model.classifier = nn.Linear(1920, nb_classes, bias=False)

            model.bottleneck.apply(weights_init_kaiming)
            model.classifier.apply(weights_init_classifier)
        else: model.classifier = nn.Linear(1920, nb_classes)
        if not pretraining and use_pretrained:
            model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))
        # model.load_state_dict(torch.load('sop_resnet_new/Stanford_paramRes_16bit_densenet201_0.0002_5.2724883734490575e-12_10_6_2.pth'))
    return model


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
