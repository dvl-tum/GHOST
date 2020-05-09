from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import time
import os
from apex import amp
import argparse
import random
import torch.nn.functional as F
import sys
import logging
import json
import copy
import PIL

from RAdam import RAdam
import gtg as gtg_module
import net
import data_utility
import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

fh = logging.FileHandler('train_reid.txt')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)


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

        self.num_classes = {'cub': 100, 'cars': 98, 'Stanford': 11318, 'Market': 751, 'cuhk03': 1367}
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


def init_args():
    dataset = 'cuhk03'
    hyperparams = Hyperparameters(dataset)
    parser = argparse.ArgumentParser(
        description='Pretraining for Person Re-ID with Group Loss')
    parser.add_argument('--dataset_name', default=dataset, type=str,
                        help='The name of the dataset')
    parser.add_argument('--oversampling', default=1, type=int,
                        help='If oversampling shoulf be used')
    parser.add_argument('--num_epochs', default=100, type=int)

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
    parser.add_argument('--pretraining', default=0, type=int,
                        help='If pretraining or fine tuning is executed')

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
                        type=int, help='Number of training epochs.')
    parser.add_argument('--temperature',
                        default=hyperparams.get_softmax_temperature(),
                        help='Temperature parameter for the softmax')'''

    parser.add_argument('--nb_workers', default=4, type=int,
                        help='Number of workers for dataloader.')
    parser.add_argument('--net_type', default='resnet50', type=str,
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
    parser.add_argument('--decrease_learning_rate', default=10., type=float,
                        help='Number to divide the learnign rate with')
    parser.add_argument('--id', default=1, type=int,
                        help='id, in case you run multiple independent nets, for example if you want an ensemble of nets')
    parser.add_argument('--is_apex', default=0, type=int,
                        help='if 1 use apex to do mixed precision training')


    return parser.parse_args()

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


def get_data_loaders(input_size, data_dir, batch_size, oversampling, train_percentage):

    with open(os.path.join(data_dir, 'splits.json'), 'r') as f:
        obj = json.load(f)[0]
    train_indices = obj['trainval']

    image_datasets = get_datasets(input_size, train_indices, data_dir, oversampling, train_percentage)

    train = torch.utils.data.DataLoader(image_datasets['train'],
                                        batch_size=batch_size, shuffle=True,
                                        num_workers=4)
    if len(image_datasets['val']) != 0:
        val = torch.utils.data.DataLoader(image_datasets['val'],
                                            batch_size=batch_size, shuffle=True,
                                            num_workers=4)
    else:
        val = None

    return train, val

def get_datasets(input_size, train_indices, data_dir, oversampling, train_percentage):

    # get samples
    train, val, labels_train, labels_val, map = get_samples(train_indices, data_dir, oversampling, train_percentage)
    import dataset.utils as utils
    # TODO: input size and mean make_transform()
    data_transforms = utils.make_transform()
    data_transforms = {'train': data_transforms, 'val': data_transforms}

    print("Initializing Datasets and Dataloaders...")
    train_dataset = DataSet(root=os.path.join(data_dir, 'images'),
                            labels=labels_train,
                            file_names=train,
                            transform=data_transforms['train'])
    val_dataset = DataSet(root=os.path.join(data_dir, 'images'),
                          labels=labels_val, file_names=val,
                          transform=data_transforms['val'])

    return {'train': train_dataset, 'val': val_dataset}

def get_samples(train_indices, data_dir, oversampling, train_percentage):
    # get train and val samples and split
    train = list()
    labels_train = list()
    val = list()
    labels_val = list()

    samps = list()
    for ind in train_indices:
        samples = os.listdir(
            os.path.join(data_dir, 'images', "{:05d}".format(ind)))
        samples = [os.path.join(os.path.join(data_dir, 'images', "{:05d}".format(ind)), samp) for samp in samples]
        samps.append(samples)

    max_num = max([len(c) for c in samps])

    random.seed(40)
    for i, samples in enumerate(samps):
        num_train = int(train_percentage * len(samples))
        train_samps = samples[:num_train]
        val_samps = samples[num_train:]

        if oversampling:
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

class PreTrainer():
    def __init__(self, args, data_dir, device, save_folder_results, save_folder_nets):
        self.device = device
        self.data_dir = data_dir
        self.args = args
        self.save_folder_results = save_folder_results
        self.save_folder_nets = save_folder_nets

    def train_model(self, config):

        file_name = 'intermediate_model'

        model = net.load_net(dataset=self.args.dataset_name, net_type=self.args.net_type,
                             nb_classes=self.args.nb_classes, embed=self.args.embed,
                             sz_embedding=self.args.sz_embedding,
                             pretraining=self.args.pretraining)
        model = model.to(self.device)

        gtg = gtg_module.GTG(self.args.nb_classes, max_iter=config['num_iter_gtg'],
                      sim=self.args.sim_type,
                      set_negative=self.args.set_negative, device=self.device).to(self.device)
        opt = RAdam(
            [{'params': list(set(model.parameters())), 'lr': config['lr']}],
            weight_decay=config['weight_decay'])
        criterion = nn.NLLLoss().to(self.device)
        criterion2 = nn.CrossEntropyLoss().to(self.device)

        # do training in mixed precision
        if self.args.is_apex:
            model, opt = amp.initialize(model, opt, opt_level="O1")

        # create loaders
        if not self.args.pretraining:
            batch_size = config['num_classes_iter'] * config['num_elements_class']
            dl_tr, dl_ev, _, _ = data_utility.create_loaders(self.args.cub_root,
                                                             self.args.nb_classes,
                                                             self.args.cub_is_extracted,
                                                             self.args.nb_workers,
                                                             config['num_classes_iter'],
                                                             config['num_elements_class'],
                                                             batch_size)
        else:
            running_corrects = 0
            batch_size = 64
            oversampling = 0
            train_percentage = 1
            input_size = 224
            dl_tr, dl_ev = get_data_loaders(input_size, self.data_dir, batch_size, oversampling, train_percentage)


        since = time.time()
        best_accuracy = 0
        scores = []
        for e in range(1, self.args.nb_epochs + 1):
            logger.info('Epoch {}/{}'.format(e, self.args.nb_epochs))
            if e == 31:
                model.load_state_dict(torch.load(
                    os.path.join(self.save_folder_nets, file_name + '.pth')))
                for g in opt.param_groups:
                    g['lr'] = config['lr'] / 10.

            if e == 51:
                model.load_state_dict(torch.load(
                    os.path.join(self.save_folder_nets, file_name + '.pth')))
                for g in opt.param_groups:
                    g['lr'] = config['lr'] / 10.

            i = 0
            for x, Y in dl_tr:
                Y = Y.to(self.device)
                opt.zero_grad()

                probs, fc7 = model(x.to(self.device))
                loss = criterion2(probs, Y)
                if not self.args.pretraining:
                    labs, L, U = data_utility.get_labeled_and_unlabeled_points(
                        labels=Y,
                        num_points_per_class=config['num_labeled_points_class'],
                        num_classes=self.args.nb_classes)

                    # compute the smoothed softmax
                    probs_for_gtg = F.softmax(probs / config['temperature'])

                    # do GTG (iterative process)
                    probs_for_gtg, W = gtg(fc7, fc7.shape[0], labs, L, U,
                                           probs_for_gtg)
                    probs_for_gtg = torch.log(probs_for_gtg + 1e-12)

                    # compute the losses
                    loss1 = criterion(probs_for_gtg, Y)
                    loss = self.args.scaling_loss * loss1 + loss
                else:
                    _, preds = torch.max(probs, 1)
                    running_corrects += torch.sum(preds == Y.data).cpu().data.item()
                i += 1

                # check possible net divergence
                if torch.isnan(loss):
                    logger.error("We have NaN numbers, closing\n\n\n")
                    sys.exit(0)

                # backprop
                if self.args.is_apex:
                    with amp.scale_loss(loss, opt) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                opt.step()

            # compute recall and NMI at the end of each epoch (for Stanford NMI takes forever so skip it)
            if not self.args.pretraining:
                with torch.no_grad():
                    logging.info("**Evaluating...**")
                    nmi, recall = utils.evaluate(model, dl_ev, self.args.nb_classes,
                                                 self.args.net_type,
                                                 dataroot=self.args.dataset_name)
                    logger.info('Recall {}, NMI {}'.format(recall, nmi))
                    scores.append((nmi, recall))
                    model.current_epoch = e
                    if recall[0] > best_accuracy:
                        best_accuracy = recall[0]
                        torch.save(model.state_dict(),
                                   os.path.join(self.save_folder_nets,
                                                file_name + '.pth'))
            else:
                logger.info('Loss {}, Recall {}'.format(torch.mean(loss.cpu()), running_corrects/len(dl_tr)))
                scores.append(running_corrects/dl_tr.shape[0])

        return scores, model

def rnd(lower, higher):
    exp = random.randint(-higher, -lower)
    base = 0.9 * random.random() + 0.1
    return base * 10 ** exp

def main():
    args = init_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    save_folder_results = 'search_results'
    save_folder_nets = 'search_results_net'
    if not os.path.isdir(save_folder_results):
        os.makedirs(save_folder_results)
    if not os.path.isdir(save_folder_nets):
        os.makedirs(save_folder_nets)

    trainer = PreTrainer(args, args.cub_root, device,
                         save_folder_results, save_folder_nets)

    best_recall = 0
    num_iter = 100
    # Random search
    for i in range(num_iter):
        logger.info('Search iteration {}'.format(i))

        # random search for hyperparameters
        lr = 10**random.uniform(-8, -3)
        batch_size = random.choice([8, 16, 32, 64])
        weight_decay = 10 ** random.uniform(-15, -6)
        num_classes_iter = random.randint(2, 5)
        num_elements_classes = random.randint(4, 9)
        num_labeled_class = random.randint(1, 3)
        decrease_lr = random.randint(0, 15)  # --> Hyperparam to search?
        set_negative = random.choice([0, 1]) # --> Hyperparam to search?
        #sim_type = random.choice(0, 1)
        num_iter_gtg = random.randint(1, 3) # --> Hyperparam to search?
        temp = random.randint(50, 80)


        config = {'lr': lr,
                  'weight_decay': weight_decay,
                  'batch_size': batch_size,
                  'num_classes_iter': num_classes_iter,
                  'num_elements_class': num_elements_classes,
                  'num_labeled_points_class': num_labeled_class,
                  'decrease_lr': decrease_lr,
                  'set_negative': set_negative,
                  'num_iter_gtg': num_iter_gtg,
                  'temperature': temp}

        hypers = ', '.join([k + ': ' + str(v) for k, v in config.items()])
        logger.info('Using Parameters: ' + hypers)

        scores, model = trainer.train_model(config)

        model = model.load_state_dict(torch.load(os.path.join(
            save_folder_nets, 'intermediate_model' + '.pth')))
        if not args.pretraining:
            recall_max = max([s[1][0] for s in scores])
        else:
            recal_max = max(scores)
        logger.info('Best Recall: {}'.format(recall_max))

        file_name = str(recall_max) + '_' + args.dataset_name + '_' + str(
            args.id) + '_' + args.net_type + '_' + str(
            config['lr']) + '_' + str(config['weight_decay']) + '_' + str(
            config['num_classes_iter']) + '_' + str(
            config['num_elements_class']) + '_' + str(
            config['num_labeled_points_class'])

        with open(os.path.join(save_folder_results, file_name + '.txt'),
                  'w') as fp:
            fp.write(file_name + "\n")
            fp.write(str(args))
            fp.write('\n')
            fp.write(str(config))
            fp.write('\n')
            fp.write('\n'.join('%s %s' % x for x in scores))
            fp.write("\n\n\n")

        if recall_max > best_recall:
            best_model = model
            best_recall = recall_max
            best_hypers = '_'.join([str(k) + '_' + str(v) for k, v in config.items()])

    if args.pretraining:
        mode = 'finetuned_'
    else:
        mode = ''
    torch.save(best_model, mode + args.model_name + '_' + args.dataset_name + '.pth')

    logger.info("Best Hyperparameters found: " + best_hypers)
    logger.info("-----------------------------------------------------\n")



if __name__ == '__main__':
    main()
