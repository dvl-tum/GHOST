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
from RAdam import RAdam


class Hyperparameters():
    def __init__(self, dataset_name='cub'):
        self.dataset_name = dataset_name
        if dataset_name == 'cub':
            self.dataset_path = '../../datasets/CUB_200_2011'
        elif dataset_name == 'cars':
            self.dataset_path = '../../datasets/CARS'
        elif dataset_name == 'Market':
            self.dataset_path = '../../datasets/Market'
        elif dataset_name == 'cuhk03':
            self.dataset_path = '../../datasets/cuhk03'
        else:
            self.dataset_path = '../../datasets/Stanford'

        self.num_classes = {'cub': 100, 'cars': 98, 'Stanford': 11318, 'Market': 651, 'cuhk03': 1267}
        self.num_classes_iteration = {'cub': 6, 'cars': 5, 'Stanford': 10, 'Market': 5, 'cuhk03': 5}
        self.num_elemens_class = {'cub': 9, 'cars': 7, 'Stanford': 6, 'Market': 7, 'cuhk03': 7}
        self.get_num_labeled_class = {'cub': 2, 'cars': 3, 'Stanford': 2, 'Market': 2, 'cuhk03': 2}
        # self.learning_rate = 0.0002
        self.learning_rate = {'cub': 0.0001563663718906821, 'cars': 0.0002, 'Stanford': 0.0006077651100709081, 'Market': 0.0002, 'cuhk03': 0.00002}
        self.weight_decay = {'cub': 6.059722614369727e-06, 'cars': 4.863656728256105e-07, 'Stanford': 5.2724883734490575e-12, 'Market': 4.863656728256105e-07, 'cuhk03': 4.863656728256105e-07}
        self.softmax_temperature = {'cub': 24, 'cars': 79, 'Stanford': 54, 'Market': 79, 'cuhk03': 79}

    def get_path(self):
        return self.dataset_path

    def get_number_classes(self):
        return self.num_classes[self.dataset_name]

    def get_number_classes_iteration(self):
        return self.num_classes_iteration[self.dataset_name]

    def get_number_elements_class(self):
        return self.num_elemens_class[self.dataset_name]

    def get_number_labeled_elements_class(self):
        return self.get_num_labeled_class[self.dataset_name]

    def get_learning_rate(self):
        return self.learning_rate[self.dataset_name]

    def get_weight_decay(self):
        return self.weight_decay[self.dataset_name]

    def get_epochs(self):
        return 70

    def get_num_gtg_iterations(self):
        return 1

    def get_softmax_temperature(self):
        return self.softmax_temperature[self.dataset_name]


class DataSet(torch.utils.data.Dataset):
    def __init__(self, root, labels, file_names, transform=None):
        # e.g., labels = range(0, 50) for using first 50 classes only
        self.labels = labels
        if transform: self.transform = transform
        self.ys, self.im_paths = [], []
        for i in file_names:
            y = int(i.split('/')[-1].split('_')[0])
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


def init_args():
    dataset = 'Market'
    hyperparams = Hyperparameters(dataset)
    parser = argparse.ArgumentParser(
        description='Pretraining for Person Re-ID with Group Loss')
    parser.add_argument('--dataset_name', default=dataset, type=str,
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
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--dataset_name', default=dataset_name, type=str,
                        help='The name of the dataset')
    parser.add_argument('--cub-root', default=hyperparams.get_path(),
                        help='Path to dataset folder')
    parser.add_argument('--cub-is-extracted', action='store_true',
                        default=True,
                        help='If `images.tgz` was already extracted, do not extract it again.' +
                             ' Otherwise use extracted data.')
    parser.add_argument('--embedding-size', default=512, type=int,
                        dest='sz_embedding', help='The embedding size')
    parser.add_argument('--nb_classes',
                        default=hyperparams.get_number_classes(), type=int,
                        help='Number of first [0, N] classes used for training and ' +
                             'next [N, N * 2] classes used for evaluating with max(N) = 100.')
    '''parser.add_argument('--num_classes_iter',
                        default=hyperparams.get_number_classes_iteration(),
                        type=int,
                        help='Number of classes in the minibatch')
    parser.add_argument('--num_elements_class',
                        default=hyperparams.get_number_elements_class(),
                        type=int,
                        help='Number of samples per each class')
    parser.add_argument('--num_labeled_points_class',
                        default=hyperparams.get_number_labeled_elements_class(),
                        type=int,
                        help='Number of labeled samples per each class')
    parser.add_argument('--lr-net', default=hyperparams.get_learning_rate(),
                        type=float, help='The learning rate')
    parser.add_argument('--weight-decay',
                        default=hyperparams.get_weight_decay(), type=float,
                        help='The l2 regularization strength')
    parser.add_argument('--nb_epochs', default=hyperparams.get_epochs(),
                        type=int, help='Number of training epochs.')'''
    parser.add_argument('--nb_workers', default=4, type=int,
                        help='Number of workers for dataloader.')
    parser.add_argument('--net_type', default='bn_inception', type=str,
                        choices=['bn_inception', 'densenet121', 'densenet161',
                                 'densenet169', 'densenet201',
                                 'resnet18', 'resnet34', 'resenet50',
                                 'resnet101', 'resnet152'],
                        help='The type of net we want to use')
    parser.add_argument('--sim_type', default='correlation', type=str,
                        help='type of similarity we want to use')
    parser.add_argument('--set_negative', default='hard', type=str,
                        help='type of threshold we want to do'
                             'hard - put all negative similarities to 0'
                             'soft - subtract minus (-) negative from each entry')
    parser.add_argument('--num_iter_gtg',
                        default=hyperparams.get_num_gtg_iterations(), type=int,
                        help='Number of iterations we want to do for GTG')
    parser.add_argument('--embed', default=0, type=int,
                        help='boolean controling if we want to do embedding or not')
    parser.add_argument('--scaling_loss', default=1, type=int,
                        dest='scaling_loss',
                        help='Scaling parameter for the loss')
    '''parser.add_argument('--temperature',
                        default=hyperparams.get_softmax_temperature(),
                        help='Temperature parameter for the softmax')'''
    parser.add_argument('--decrease_learning_rate', default=10., type=float,
                        help='Number to divide the learnign rate with')
    parser.add_argument('--id', default=1, type=int,
                        help='id, in case you run multiple independent nets, for example if you want an ensemble of nets')
    parser.add_argument('--is_apex', default=0, type=int,
                        help='if 1 use apex to do mixed precision training')


    return parser.parse_args()


class PreTrainer():
    def __init__(self, args, data_dir, save_name, device):
        self.device = device
        self.data_dir = data_dir
        self.save_name = save_name
        self.args = args

    def train_model(self, config):

        # Number of classes in the dataset
        with open(os.path.join(self.data_dir, 'splits.json'), 'r') as f:
            obj = json.load(f)[0]
        train_indices = obj['trainval']
        num_classes = len(train_indices)

        model, input_size, params_to_update = self.get_model(num_classes)

        optimizer = RAdam([{'params': params_to_update, 'lr': config['lr']}], weight_decay=config['weight_decay'])

        if self.args.apex_on:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

        criterion = nn.CrossEntropyLoss()

        dataloaders = self.get_data_loaders(input_size, train_indices, config['batch_size'])

        since = time.time()
        val_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(self.args.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.args.num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                epoch_loss, epoch_acc = self.run_model(phase, model, dataloaders,
                                                  optimizer, criterion)
                # if phase == 'val':
                #    tune.track.log(mean_accuracy=epoch_acc)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss,
                                                           epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), self.save_name)
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

        time_elapsed = time.time() - since
        print('Model complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                            time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)

        return model, val_acc_history

    def run_model(self, phase, model, dataloaders, optimizer, criterion):

        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                if (self.args.model_name == 'inception') and phase == 'train':
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
                    if self.args.apex_on:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            try:
                                scaled_loss.backward()
                            except:
                                return 100, 0
                    else:
                        loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

        return epoch_loss, epoch_acc

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def get_model(self, num_classes):

        model, input_size = self.initialize_model(num_classes,
                                                  use_pretrained=True)
        model = model.to(self.device)
        # params to fine tune
        params_to_update = model.parameters()
        print("Params to learn:")
        if self.args.feature_extract:
            params_to_update = []
            for name, param in model_ft.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)

        return model, input_size, params_to_update

    def initialize_model(self, num_classes,
                         use_pretrained=True):

        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0

        if self.args.model_name == "resnet":
            """ Resnet50            
            """
            model_ft = models.resnet50(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.args.feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif self.args.model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.args.feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif self.args.model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.args.feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif self.args.model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.args.feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes,
                                               kernel_size=(1, 1),
                                               stride=(1, 1))
            model_ft.num_classes = num_classes
            input_size = 224

        elif self.args.model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet121(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.args.feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif self.args.model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.args.feature_extract)
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

    def get_datasets(self, input_size, train_indices):

        # get samples
        train, val, labels_train, labels_val, map = self.get_samples(
            train_indices)

        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomAffine(degrees=0, scale=(256, 480)),
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.RandomAffine(degrees=0, scale=(256, 480)),
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
        }

        print("Initializing Datasets and Dataloaders...")
        train_dataset = DataSet(root=os.path.join(self.data_dir, 'images'),
                                labels=labels_train,
                                file_names=train,
                                transform=data_transforms['train'])
        val_dataset = DataSet(root=os.path.join(self.data_dir, 'images'),
                              labels=labels_val, file_names=val,
                              transform=data_transforms['val'])

        return {'train': train_dataset, 'val': val_dataset}

    def get_data_loaders(self, input_size, train_indices, batch_size):

        image_datasets = self.get_datasets(input_size, train_indices)

        dataloaders_dict = {
            x: torch.utils.data.DataLoader(image_datasets[x],
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=4) for x in['train']}

        return dataloaders_dict

    def get_samples(self, train_indices):
        # get train and val samples and split
        train = list()
        val = list()
        labels_train = list()
        labels_val = list()

        train_percentage = 1
        samps = list()
        for ind in train_indices:
            samples = os.listdir(
                os.path.join(self.data_dir, 'images', "{:05d}".format(ind)))
            samples = [os.path.join(
                os.path.join(self.data_dir, 'images', "{:05d}".format(ind)), samp)
                for
                samp in samples]
            samps.append(samples)
        max_num = max([len(c) for c in samps])

        random.seed(40)
        for i, samples in enumerate(samps):
            num_train = int(train_percentage * len(samples))
            train_samps = samples[:num_train]
            val_samps = samples[num_train:]

            if self.args.oversampling:
                choose_train = copy.deepcopy(train_samps)
                while len(train_samps) < int(max_num * train_percentage):
                    train_samps += [random.choice(choose_train)]

                choose_val = copy.deepcopy(val_samps)
                while len(val_samps) < int(max_num * (1 - train_percentage)):
                    val_samps += [random.choice(choose_val)]

                for v in val_samps:
                    if v in train_samps:
                        print(
                            "Sample of validation set in training set - End.")
                        quit()

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

def rnd(lower, higher):
    exp = random.randint(-higher, -lower)
    base = 0.9 * random.random() + 0.1
    return base * 10 ** exp

def main():
    args = init_args()
    data_dir = os.path.join('../../datasets', args.dataset_name)
    save_name = args.model_name + '_' + args.dataset_name + '_pretrained.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir('search_results'):
        os.makedirs('search_results')

    trainer = PreTrainer(args, data_dir, save_name, device)

    best_acc = 0
    num_iter = 100
    # Random search
    for i in range(num_iter):

        # random search for hyperparameters
        lr = 10**random.uniform(-8, -3)
        batch_size = random.choice([8, 16, 32, 64])
        weight_decay = 10 ** random.uniform(-15, -6)

        config = {'lr': lr,
                  'weight_decay': weight_decay,
                  'batch_size': batch_size}

        model, val_acc_history = trainer.train_model(config)

        hypers = '_'.join([str(k) + '_' + str(v) for k, v in config.items()])

        acc = max(val_acc_history)
        if acc > best_acc:
            best_model = model
            best_acc = acc
            best_hypers = hypers

        save_name = os.path.join('search_results', str(acc) + hypers + '.txt')
        with open(save_name, 'w') as file:
            for e in val_acc_history:
                file.write(str(e.data))
                file.write('\n')

    torch.save(best_model, 'fine_tuned' + args.model_name + args.dataset_name + '.pth')

    print("Best Hyperparameters found: " + best_hypers)



if __name__ == '__main__':
    main()
