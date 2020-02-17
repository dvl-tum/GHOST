import argparse
import random


def rnd(lower, higher):
    exp = random.randint(-higher, -lower)
    base = 0.9 * random.random() + 0.1
    return base * 10 ** exp


class Hyperparameters():
    def __init__(self, dataset_name='cub'):
        self.dataset_name = dataset_name
        if dataset_name == 'cub':
            self.dataset_path = '../gtg_embedding/CUB_200_2011'
        elif dataset_name == 'cars':
            self.dataset_path = '../gtg_embedding/CUB_200_2011'
        else:
            self.dataset_path = '../gtg_embedding/Stanford'

    def get_path(self):
        return self.dataset_path

    def get_number_classes(self):
        if self.dataset_name == 'cub':
            return 100
        elif self.dataset_name == 'cars':
            return 98
        else:
            return 100

    def get_number_classes_iteration(self):
        return random.randint(5, 10)

    def get_number_elements_class(self):
        return random.randint(5, 10)

    def get_number_labeled_elements_class(self):
        return random.randint(1, 4)

    def get_learning_rate(self):
        return rnd(2, 4)

    def get_weight_decay(self):
        return rnd(5, 15)

    def get_epochs(self):
        return 3

    def get_num_gtg_iterations(self):
        return random.randint(1, 3)

    def get_softmax_temperature(self):
        return random.randint(1, 100)


parser = argparse.ArgumentParser(description='Training inception V2' +
                                             ' (BNInception) on CUB200 with Proxy-NCA loss as described in ' +
                                             '`No Fuss Distance Metric Learning using Proxies.`')
dataset_name = 'cub'  # cub, cars or stanford
parser.add_argument('--dataset_name', default=dataset_name, type=str, help='The name of the dataset')
hyperparams = Hyperparameters(dataset_name)
parser.add_argument('--cub-root', default=hyperparams.get_path(), help='Path to dataset folder')
parser.add_argument('--cub-is-extracted', action='store_true',
                    default=True, help='If `images.tgz` was already extracted, do not extract it again.' +
                                       ' Otherwise use extracted data.')
parser.add_argument('--embedding-size', default=512, type=int, dest='sz_embedding', help='The embedding size')
parser.add_argument('--nb_classes', default=hyperparams.get_number_classes(), type=int,
                    help='Number of first [0, N] classes used for training and ' +
                         'next [N, N * 2] classes used for evaluating with max(N) = 100.')
parser.add_argument('--num_classes_iter', default=hyperparams.get_number_classes_iteration(), type=int,
                    help='Number of classes in the minibatch')
parser.add_argument('--num_elements_class', default=hyperparams.get_number_elements_class(), type=int,
                    help='Number of samples per each class')
parser.add_argument('--num_labeled_points_class', default=hyperparams.get_number_labeled_elements_class(), type=int,
                    help='Number of labeled samples per each class')
parser.add_argument('--lr-net', default=hyperparams.get_learning_rate(), type=float, help='The learning rate')
parser.add_argument('--weight-decay', default=hyperparams.get_weight_decay(), type=float, help='The l2 regularization strength')
parser.add_argument('--nb_epochs', default=hyperparams.get_epochs(), type=int, help='Number of training epochs.')
parser.add_argument('--nb_workers', default=8, type=int, help='Number of workers for dataloader.')
parser.add_argument('--net_type', default='bn_inception', type=str, help='The type of net we want to use')
parser.add_argument('--sim_type', default='correlation', type=str, help='type of similarity we want to use')
parser.add_argument('--set_negative', default='soft', type=str,
                    help='type of threshold we want to do'
                         'hard - put all negative similarities to 0'
                         'soft - subtract minus (-) negative from each entry')
parser.add_argument('--num_iter_gtg', default=hyperparams.get_num_gtg_iterations(), type=int, help='Number of iterations we want to do for GTG')
parser.add_argument('--embed', default=True, type=bool, help='boolean controling if we want to do embedding or not')
parser.add_argument('--use_prior', default=True, type=bool, help='if True use prior otherwise not')
parser.add_argument('--use_double_loss', default=True, type=bool, dest='use_double_loss', help='if True use also classification loss')
parser.add_argument('--scaling_loss', default=1, type=int, dest='scaling_loss', help='Scaling parameter for the loss')
parser.add_argument('--temperature', default=hyperparams.get_softmax_temperature(), help='Temperature parameter for the softmax')
parser.add_argument('--decrease_learning_rate', default=10., type=float,
                    help='Number of iterations we want to do for GTG')
parser.add_argument('--evaluate_beginning', default=False, type=bool, help='if True evaluate at the beginning')
parser.add_argument('--revert_best_accuracy', default=True, type=bool,
                    help='if True, at the end of each epoch revert the net to the checkpoint with the best accuracy')
