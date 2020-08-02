class Hyperparameters():
    def __init__(self, dataset_name='cub', net_type='densenet161'):
        self.dataset_name = dataset_name
        self.net_type = net_type

        # print('Without GL')
        if dataset_name == 'Market' or dataset_name == 'MarketCuhk03':
            self.dataset_path = '../../datasets/Market-1501-v15.09.15'
            self.dataset_short = 'Market'
        elif dataset_name == 'cuhk03-detected':
            self.dataset_path = '../../datasets/cuhk03/detected'
            self.dataset_short = 'cuhk03'
        elif dataset_name == 'cuhk03-labeled':
            self.dataset_path = '../../datasets/cuhk03/labeled'
            self.dataset_short = 'cuhk03'
        elif dataset_name == 'cuhk03-np-detected':
            self.dataset_path = '../../datasets/cuhk03-np/labeled'
            self.dataset_short = 'cuhk03-np'
        elif dataset_name == 'cuhk03-np-labeled':
            self.dataset_path = '../../datasets/cuhk03-np/labeled'
            self.dataset_short = 'cuhk03-np'
        elif dataset_name == 'dukemtmc':
            self.dataset_path = '../../datasets/dukemtmc'
            self.dataset_short = 'dukemtmc'

        self.lamd = {'Market': 0.3, 'cuhk03-detected': 0.3, 'dukemtmc': 0.3}

        self.k1 = {'Market': 20, 'cuhk03-detected': 20, 'dukemtmc': 20}

        self.k2 = {'Market': 6, 'cuhk03-detected': 6, 'dukemtmc': 6}

        self.num_classes = {'Market': 751,
                            'cuhk03-detected': 1367,
                            'cuhk03-np-detected': 767,
                            'cuhk03-labeled': 1367,
                            'cuhk03-np-labeled': 767,
                            'dukemtmc': 702,
                            'MarketCuhk03': 751 + 1367}
        self.num_classes_iteration = {
            'Market': {'resnet50': 3, 'densenet161': 5},
            'cuhk03-detected': {'resnet50': 5, 'densenet161': 5},
            'cuhk03-np-detected': {'resnet50': 5, 'densenet161': 5},
            'cuhk03-labeled': {'resnet50': 5, 'densenet161': 5},
            'cuhk03-np-labeled': {'resnet50': 5, 'densenet161': 5},
            'dukemtmc': {'resnet50': 5, 'densenet161': 5},
            'MarketCuhk03': {'resnet50': 5, 'densenet161': 5}}

        self.num_elemens_class = {'Market': {'resnet50': 4, 'densenet161': 6},
                                  'cuhk03-detected': {'resnet50': 5,
                                                      'densenet161': 8},
                                  'cuhk03-np-detected': {'resnet50': 5,
                                                         'densenet161': 8},
                                  'cuhk03-labeled': {'resnet50': 5,
                                                     'densenet161': 8},
                                  'cuhk03-np-labeled': {'resnet50': 5,
                                                        'densenet161': 8},
                                  'dukemtmc': {'resnet50': 5,
                                               'densenet161': 8},
                                  'MarketCuhk03': {'resnet50': 5,
                                                   'densenet161': 8}}

        self.get_num_labeled_class = {
            'Market': {'resnet50': 3, 'densenet161': 1},
            'cuhk03-detected': {'resnet50': 2, 'densenet161': 1},
            'cuhk03-np-detected': {'resnet50': 2, 'densenet161': 1},
            'cuhk03-labeled': {'resnet50': 2, 'densenet161': 1},
            'cuhk03-np-labeled': {'resnet50': 2, 'densenet161': 1},
            'dukemtmc': {'resnet50': 2, 'densenet161': 1},
            'MarketCuhk03': {'resnet50': 2, 'densenet161': 1}}

        self.learning_rate = {'Market': {'resnet50': 1.289377564403867e-05,
                                         'densenet161': 8.201555304285775e-05},
                              'cuhk03-detected': {
                                  'resnet50': 4.4819286767613e-05,
                                  'densenet161': 6.938966913758872e-05},
                              'cuhk03-np-detected': {
                                  'resnet50': 4.4819286767613e-05,
                                  'densenet161': 6.938966913758872e-05},
                              'cuhk03-labeled': {
                                  'resnet50': 4.4819286767613e-05,
                                  'densenet161': 6.938966913758872e-05},
                              'cuhk03-np-labeled': {
                                  'resnet50': 4.4819286767613e-05,
                                  'densenet161': 6.938966913758872e-05},
                              'dukemtmc': {'resnet50': 4.4819286767613e-05,
                                           'densenet161': 6.938966913758872e-05},
                              'MarketCuhk03': {'resnet50': 4.4819286767613e-05,
                                               'densenet161': 6.938966913758872e-05}}

        self.weight_decay = {'Market': {'resnet50': 1.9250447877921047e-14,
                                        'densenet161': 4.883141881206216e-11},
                             'cuhk03-detected': {
                                 'resnet50': 1.5288509425482333e-13,
                                 'densenet161': 1.6553076469649952e-07},
                             'cuhk03-np-detected': {
                                 'resnet50': 1.5288509425482333e-13,
                                 'densenet161': 1.6553076469649952e-07},
                             'cuhk03-labeled': {
                                 'resnet50': 1.5288509425482333e-13,
                                 'densenet161': 1.6553076469649952e-07},
                             'cuhk03-np-labeled': {
                                 'resnet50': 1.5288509425482333e-13,
                                 'densenet161': 1.6553076469649952e-07},
                             'dukemtmc': {'resnet50': 1.5288509425482333e-13,
                                          'densenet161': 1.6553076469649952e-07},
                             'MarketCuhk03': {
                                 'resnet50': 1.5288509425482333e-13,
                                 'densenet161': 1.6553076469649952e-07}}

        self.softmax_temperature = {
            'Market': {'resnet50': 80, 'densenet161': 37},
            'cuhk03-detected': {'resnet50': 80, 'densenet161': 34},
            'cuhk03-np-detected': {'resnet50': 80, 'densenet161': 34},
            'cuhk03-labeled': {'resnet50': 80, 'densenet161': 34},
            'cuhk03-np-labeled': {'resnet50': 80, 'densenet161': 34},
            'dukemtmc': {'resnet50': 80, 'densenet161': 34},
            'MarketCuhk03': {'resnet50': 80, 'densenet161': 34}}

        self.num_iter_gtg = {'Market': {'resnet50': 2, 'densenet161': 3},
                             'cuhk03-detected': {'resnet50': 1,
                                                 'densenet161': 2},
                             'cuhk03-np-detected': {'resnet50': 1,
                                                    'densenet161': 2},
                             'cuhk03-labeled': {'resnet50': 1,
                                                'densenet161': 2},
                             'cuhk03-np-labeled': {'resnet50': 1,
                                                   'densenet161': 2},
                             'dukemtmc': {'resnet50': 1, 'densenet161': 2},
                             'MarketCuhk03': {'resnet50': 1, 'densenet161': 2}}

    def get_dataset_path(self):
        return self.dataset_path

    def get_num_classes(self):
        return self.num_classes[self.dataset_name]

    def get_number_classes_iteration(self):
        return self.num_classes_iteration[self.dataset_name][self.net_type]

    def get_number_elements_class(self):
        return self.num_elemens_class[self.dataset_name][self.net_type]

    def get_number_labeled_elements_class(self):
        return self.get_num_labeled_class[self.dataset_name][self.net_type]

    def get_learning_rate(self):
        return self.learning_rate[self.dataset_name][self.net_type]

    def get_weight_decay(self):
        return self.weight_decay[self.dataset_name][self.net_type]

    def get_nb_epochs(self):
        return 70

    def get_num_gtg_iterations(self):
        return self.num_iter_gtg[self.dataset_name][self.net_type]

    def get_softmax_temperature(self):
        return self.softmax_temperature[self.dataset_name][self.net_type]

    def get_rerank_lambda(self):
        return self.lamb[self.dataset_name]

    def get_rerank_k1(self):
        return self.k1[self.dataset_name]

    def get_rerank_k2(self):
        return self.k2[self.dataset_name]