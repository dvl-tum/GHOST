from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import json
import dataset
import PIL
from apex import amp
import argparse
import copy
import random
#from ray import tune
#from ray.tune.examples.mnist_pytorch import get_data_loaders, ConvNet, train, test


def init_args():
    parser = argparse.ArgumentParser(
        description='Pretraining for Person Re-ID with Group Loss')
    parser.add_argument('--dataset_name', default='Market', type=str,
                        help='The name of the dataset')
    parser.add_argument('--apex_on', default=1, type=int,
                        help='If apex should be used')
    parser.add_argument('--oversampling', default=1, type=int,
                        help='If oversampling shoulf be used')
    parser.add_argument('--manually_lr_decay', default=0, type=int,
                        help='If manually lr decay should be applied')
    parser.add_argument('--model_name', default='resnet', type=str,
                        help='which model shoul be fine tuned')
    parser.add_argument('--feature_extract', default=0, type=int,
                        help='If only last layer should be updated')

    return parser.parse_args()


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25,
                is_inception=False, save_name='finetuned.pth',
                manually_lr_decay=0, apex_on=1):
    since = time.time()
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        if manually_lr_decay:
            if len(val_acc_history) > 2:
                check = [i.cpu().data.numpy() for i in val_acc_history[-2:]]
                print(np.abs(check[0] - check[1]))
                if np.abs(check[0] - check[1]) < 0.1:
                    print("Adapt LR")
                    model.load_state_dict(torch.load(save_name))
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr'] / 10.

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        if apex_on:
                            with amp.scale_loss(loss,
                                                optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(
                dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss,
                                                       epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), save_name)
            if phase == 'val':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                        time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract,
                     use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet50            
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes,
                                           kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


class DataSet(torch.utils.data.Dataset):
    def __init__(self, root, labels, file_names, transform=None):
        # e.g., labels = range(0, 50) for using first 50 classes only
        self.labels = labels
        if transform: self.transform = transform
        self.ys, self.im_paths = [], []
        for i in file_names:
            y = i.split('/')[-1].split('_')[0]
            y = int(y.strip("0"))
            # fn needed for removing non-images starting with '._'
            fn = os.path.basename(i)
            if y in self.labels and fn[:2] != '._':
                self.ys += [y]
                self.im_paths.append(i)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        im = PIL.Image.open(self.im_paths[index])
        im = self.transform(im)
        return im, self.ys[index]


def get_samples(data_dir, oversampling, train_indices):
    # get train and val samples and split
    train = list()
    val = list()
    labels_train = list()
    labels_val = list()

    train_percentage = 0.6
    samps = list()
    for ind in train_indices:
        samples = os.listdir(
            os.path.join(data_dir, 'images', "{:05d}".format(ind)))
        samples = [os.path.join(
            os.path.join(data_dir, 'images', "{:05d}".format(ind)), samp) for
            samp in samples]
        samps.append(samples)
    max_num = max([len(c) for c in samps])

    random.seed(40)
    for i, samples in enumerate(samps):
        num_train = int(train_percentage * len(samples))
        train_samps = samples[:num_train]
        val_samps = samples[num_train:]

        if oversampling:
            choose_train = copy.deepcopy(train_samps)
            while len(train_samps) < int(max_num*train_percentage):
                train_samps += [random.choice(choose_train)]

            choose_val = copy.deepcopy(val_samps)
            while len(val_samps) < int(max_num * (1 - train_percentage)):
                val_samps += [random.choice(choose_val)]

        train.append(train_samps)
        val.append(val_samps)

        labels_train.append([train_indices[i]] * len(train_samps))
        labels_val.append([train_indices[i]] * len(val_samps))

    train = [t for classes in train for t in classes]
    val = [t for classes in val for t in classes]

    # mapping of labels from 0 to num_classes
    map = {class_ind: i for i, class_ind in enumerate(train_indices)}
    labels_train = [map[t] for classes in labels_train for t in classes]
    labels_val = [map[t] for classes in labels_val for t in classes]


    return train, val, labels_train, labels_val, map


if __name__ == '__main__':
    args = init_args()
    data_dir = os.path.join('../../datasets', args.dataset_name)

    # Number of classes in the dataset
    with open(os.path.join(data_dir, 'splits.json'), 'r') as f:
        obj = json.load(f)[0]
    train_indices = obj['trainval']
    num_classes = len(train_indices)

    # get samples
    train, val, labels_train, labels_val, map = get_samples(data_dir,
                                                            args.oversampling,
                                                            train_indices)

    # hyperparams
    batch_size = 64
    num_epochs = 100
    lr = 0.001
    momentum = 0.9
    weight_decay = 0.0001
    trans = 'RandomResizedCrop, RandomHorizontalFlip, Normalization (0.485, 0.456, 0.406) (0.229, 0.224, 0.225)'
    # Initialize the modelrun
    model_ft, input_size = initialize_model(args.model_name, num_classes,
                                            args.feature_extract,
                                            use_pretrained=True)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    print("Initializing Datasets and Dataloaders...")
    train_dataset = DataSet(root=os.path.join(data_dir, 'images'),
                            labels=labels_train,
                            file_names=train,
                            transform=data_transforms['train'])
    val_dataset = DataSet(root=os.path.join(data_dir, 'images'),
                          labels=labels_val, file_names=val,
                          transform=data_transforms['val'])

    image_datasets = {'train': train_dataset, 'val': val_dataset}
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x],
                                       batch_size=batch_size, shuffle=True,
                                       num_workers=4) for x in
        ['train', 'val']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = model_ft.to(device)

    # params to fine tune
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if args.feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                # print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                pass
                # print("\t", name)

    optimizer_ft = optim.SGD(params_to_update, lr=lr, momentum=momentum,
                             weight_decay=weight_decay)
    if args.apex_on:
        model_ft, optimizer_ft = amp.initialize(model_ft, optimizer_ft,
                                                opt_level="O1")
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    print(
        'Batch size {}, momentum {}, weight decay {}, lr {}, num_epochs {}, transforms {}, manually adapt {}'.format(
            batch_size, momentum, weight_decay, lr, num_epochs, trans,
            args.manually_lr_decay))
    save_name = args.model_name + '_' + args.dataset_name + '_pretrained.pth'
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion,
                                 optimizer_ft, num_epochs=num_epochs,
                                 is_inception=(args.model_name == "inception"),
                                 save_name=save_name,
                                 manually_lr_decay=args.manually_lr_decay,
                                 apex_on=args.apex_on)
