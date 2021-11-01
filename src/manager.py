import ReID
import os.path as osp

from .datasets.Det4ReIDDataset import Det4ReIDDataset
from .nets.proxy_gen import ProxyGenMLP, ProxyGenRNN
import os
from torch.utils.data import DataLoader
import torch
from collections import defaultdict
from src.datasets.utils import collate_test, collate_train
from data.splits import _SPLITS
from .tracker import Tracker
#from .tracker_sep_act_inact import Tracker
from src.datasets.MOTDataset import MOTDataset
from src.datasets.TrackingDataset import TrackingDataset
import time
import random
import logging
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path
import sys
import argparse
#sys.path.insert(0, "/usr/wiss/seidensc/Documents/SRGAN-PyTorch")
#from srgan_pytorch.utils.common import configure

'''sys.path.insert(0, "/usr/wiss/seidensc/Documents/batch-dropblock-network")
from get_model import get_bdb
sys.path.insert(0, "/usr/wiss/seidensc/Documents/reid-strong-baseline")
from get_model_bot import get_bot
sys.path.insert(0, "/usr/wiss/seidensc/Documents/LUPerson/fast-reid")
from load_model import load_lup
sys.path.insert(0, "/usr/wiss/seidensc/Documents/LightMBN")
from load_model_lmbn import load_lightweight_mbn
sys.path.insert(0, '/usr/wiss/seidensc/Documents/TransReID')
from load_model_transreid import load_trans_reid
sys.path.insert(0, '/usr/wiss/seidensc/Documents/ABD-Net')
from get_model_abd import get_model_abd
sys.path.insert(0, '/usr/wiss/seidensc/Documents/deep-person-reid')
from get_model_os import get_model_os'''
sys.path.insert(0, '/usr/wiss/seidensc/Documents/mot_neural_solver')
from get_detector import get_detection_model
'''
from torchreid import models
import torchreid'''


logger = logging.getLogger('AllReIDTracker.Manager')


class Manager():
    def __init__(self, device, timer, dataset_cfg, reid_net_cfg, 
                tracker_cfg, train=False):
        self.device = device
        self.reid_net_cfg = reid_net_cfg
        self.dataset_cfg = dataset_cfg
        self.tracker_cfg = tracker_cfg

        #load ReID net
        self.num_classes = 10
        self.loaders = self._get_loaders(dataset_cfg)
        self._get_models()
        

        if 'train' in self.loaders.keys():
            self.num_classes = self.loaders['train'].dataset.id
            self.num_iters = 30 if self.reid_net_cfg['mode'] == 'hyper_search' else 1
        else:
            self.num_classes = self.reid_net_cfg['trained_on']['num_classes']
        
        # name of eval criteria for logging
        self.eval1 = 'MOTA'
        self.eval2 = 'IDF1'

        if self.reid_net_cfg['mode'] == 'hyper_search':
            self.num_iters = 30
        else:
            self.num_iters = 1
        
    def train(self):
        for i in range(self.num_iters):
            if self.num_iters > 1:    
                logger.info("Search iteration {}/{}".format(i, self.num_iters))
            self._setup_training()
            best_mota_iter, best_idf1_iter = 0, 0
            for e in range(self.reid_net_cfg['train_params']['epochs']):
                logger.info("Epoch {}/{}".format(e, self.reid_net_cfg['train_params']['epochs']))
                self._train(e)
                mota_overall, idf1_overall = self._evaluate()
                self._check_best(mota_overall, idf1_overall)
                if mota_overall > best_mota_iter:
                    best_mota_iter = mota_overall
                    best_idf1_iter = idf1_overall
            logger.info("Iteration {}: Best {} {} and {} {}".format(i, self.eval1, best_mota_iter, self.eval2, best_idf1_iter))

        # save best
        logger.info("Overall Results: Best {} {} and {} {}".format(self.eval1, self.best_mota, self.eval2, self.best_idf1))
        self._save_best()

    def _train(self, e):
        temp = self.reid_net_cfg['train_params']['temperature']
        losses = defaultdict(list)
        
        for data in self.loaders['train']:
            imgs = data[0]
            labels = data[1]

            if e == 31 or e == 51:
                self._reduce_lr()
            self.optimizer.zero_grad()

            labels = [{k: torch.from_numpy(v).to(self.device) for k, v in labs.items()} for labs in labels]
            imgs = [img.to(self.device) for img in imgs]

            preds1, feats1 = self.encoder(imgs, labels)

            # cross entopy loss
            if self.loss1:
                loss = self.loss1(preds1/temp, Y)
                losses['Loss 1'].append(loss)
            
            if self.gnn:
                edge_attr, edge_ind, feats1 = self.graph_gen.get_graph(feats1)
                preds2, feats2 = self.gnn(feats1, edge_ind, edge_attr, 'plain')
                if self.loss2:
                    loss2 = self.loss2(preds2[-1]/temp, Y)
                    losses['Loss 2'].append(loss2)
                    loss += loss2
            
            losses['Total'].append(loss)
            loss.backward()
            self.optimizer.step()

        logger.info({k: sum(v)/len(v) for k, v in losses.items()})  

    def _evaluate(self, mode='val'):
        names = list()
        corresponding_gt = OrderedDict()
        
        # get tracking files
        print(self.loaders)
        for seq in self.loaders[mode]:
            if self.dataset_cfg['splits'] != 'mot17_test':
                df = seq[0].corresponding_gt
                df = df.drop(['bb_bot', 'bb_right'], axis=1)
                df = df.rename(columns={"frame": "FrameId", "id": "Id", "bb_left": "X", "bb_top": "Y", "bb_width": "Width", "bb_height": "Height", "conf": "Confidence", "label": "ClassId", "vis": "Visibility"})
                df = df.set_index(['FrameId', 'Id'])
                corresponding_gt[seq[0].name] = df
            self.tracker.gnn, self.tracker.encoder = self.gnn, self.encoder
            self.tracker.track(seq[0])
            names.append(seq[0].name)

        #self.tracker.experiment = 'TMOH' # "center_track"

        # get tracking results
        if self.dataset_cfg['splits'] != 'mot17_test':
            # evlauate only with gt files corresponding to detection files
            # --> no FN here
            if self.tracker_cfg['oracle']:
                accs, names = self._get_results(names, corresponding_gt)
                self._get_summary(accs, names, "Oracle GT")

                # Oracle Output --> What was this again?!
                if self.dataset_cfg['save_oracle']:
                    accs, names = self._get_results(names, get_gt_files=True)
                    self._get_summary(accs, names, "Oracle results")

            # Evaluate files from tracking
            accs, names = self._get_results(names)
            self._get_summary(accs, names, "Normal")

        return None, None 

    def _get_models(self):
        tracker = Tracker
        test = 1 if self.dataset_cfg['splits'] == 'mot17_test' else 0

        self._get_encoder()
        self._get_sr_gan()
        self._get_proxy_gen()

        if self.reid_net_cfg['gnn']:
            self._get_gnn()
            self.tracker = tracker(self.tracker_cfg, self.encoder, 
                                self.gnn, self.graph_gen, 
                                self.proxy_gen, dev=self.device,
                                net_type=self.net_type,
                                test=test, sr_gan=self.sr_gan,
                                output=self.reid_net_cfg['output'])
            self.tracker.num_el_id = self.reid_net_cfg['dl_params']['num_elements_class']
        else:
            self.gnn, self.graph_gen = None, None
            self.tracker = tracker(self.tracker_cfg, self.encoder, 
                                proxy_gen=self.proxy_gen, dev=self.device, 
                                net_type=self.net_type,
                                test=test, sr_gan=self.sr_gan,
                                output=self.reid_net_cfg['output'])

    def _get_proxy_gen(self):
        # currently no proxy gen implemented
        self.proxy_gen = None

    def _get_sr_gan(self):
        # currently no SR GAN implemented
        self.sr_gan = None

    def _get_encoder(self):
        self.net_type = self.reid_net_cfg['encoder_params']['net_type']
        if self.net_type == 'batch_drop_block':
            # get pretrained batch drop block network
            encoder = get_bdb()
            self.encoder = encoder.to(self.device)
            self.sz_embed = None
        
        elif self.net_type == 'LUP':
            encoder = load_lup()
            self.encoder = encoder.to(self.device)
            self.sz_embed = None

        elif self.net_type == 'abd':
            encoder = get_model_abd()
            self.encoder = encoder.to(self.device)
            self.sz_embed = None

        elif self.net_type == 'transreid':
            encoder = load_trans_reid()
            self.encoder = encoder.to(self.device)
            self.sz_embed = None

        elif self.net_type == 'lmbn':
            encoder = load_lightweight_mbn()
            self.encoder = encoder.to(self.device)
            self.sz_embed = None

        elif self.net_type == 'os':
            encoder = get_model_os()
            self.encoder = encoder.to(self.device)
            self.sz_embed = None

        elif self.net_type == 'bot':
            # get pretraine bag of tricks network
            encoder = get_bot()
            self.encoder = encoder.to(self.device)
            self.sz_embed = None
        
        elif self.net_type == 'det_bb':
            encoder = get_detection_model()
            self.encoder = encoder.to(self.device)
            self.sz_embed = None

        elif self.net_type == 'resnet50_analysis':
            # get pretrained resnet 50 from torchreid
            encoder = torchreid.models.build_model(name='resnet50', num_classes=1000)
            torchreid.utils.load_pretrained_weights(encoder, 'resnet50_market_xent.pth.tar') 
            self.sz_embed = None
        
        elif self.net_type == 'resnet50_attention':
            import copy
            params = copy.deepcopy(self.reid_net_cfg['encoder_params'])
            params['net_type'] = 'resnet50'
            encoder, self.sz_embed = ReID.net.load_net(
                self.reid_net_cfg['trained_on']['name'],
                self.num_classes, 'test',
                attention=True,
                **params)

        else:
            # own trained network
            encoder, self.sz_embed = ReID.net.load_net(
                self.reid_net_cfg['trained_on']['name'],
                self.num_classes, 'test',
                attention=False,
                **self.reid_net_cfg['encoder_params'])

        self.encoder = encoder.to(self.device)
    
    def _get_gnn(self):
        # currently no GNN implemented
        # self.reid_net_cfg['gnn_params']['classifier']['num_classes'] = self.num_classes
        self.gnn = None
        in_channels = self.sz_embed[0]
        self.graph_gen = None

        self.gnn = ReID.net.Query_Guided_Attention_Layer(in_channels, \
            gnn_params=self.reid_net_cfg['gnn_params']['gnn'],
            num_classes=self.reid_net_cfg['gnn_params']['classifier']['num_classes'],
            non_agg=True, class_non_agg=True,
            neck=self.reid_net_cfg['gnn_params']['classifier']['neck']).cuda(self.device)

        if self.reid_net_cfg['gnn_params']['pretrained_path'] != "no":
            load_dict = torch.load(
                self.reid_net_cfg['gnn_params']['pretrained_path'],
                map_location='cpu')
            load_dict = {k: v for k, v in load_dict.items() if 'fc' not in k.split('.')}

            model_dict = self.gnn.state_dict()
            model_dict.update(load_dict)
            self.gnn.load_state_dict(model_dict)

    
    def _get_loaders(self, dataset_cfg):
        # Initialize datasets
        loaders = dict()
        # train / test / eval mode?
        for mode in _SPLITS[dataset_cfg['splits']].keys():
            seqs = _SPLITS[dataset_cfg['splits']][mode]['seq']
            self.dir = _SPLITS[dataset_cfg['splits']][mode]['dir']

            # tracking dataset
            if mode != 'train':
                dataset = TrackingDataset(dataset_cfg['splits'], seqs, 
                                            dataset_cfg, self.dir, dev=self.device)
                loaders[mode] = dataset
            # train dataset
            else:
                dataset = MOTDataset(dataset_cfg['splits'], seqs, 
                                        dataset_cfg, self.dir) 
                loaders[mode] = DataLoader(dataset, batch_size=1, shuffle=False,
                                        collate_fn=collate_train)
        return loaders
    
    def _sample_params(self):
        # sample parameters for random search
        config = {'lr': 10 ** random.uniform(-5, -3),
                  'weight_decay': 10 ** random.uniform(-15, -6),
                  'temperatur': random.random(),
                  'epochs': 10}
        self.reid_net_cfg['train_params'].update(config)
        
        if self.reid_net_cfg['gnn_params']['pretrained_path'] == 'no':
            # currently no GNN implemented, but don't forget to update
            self.reid_net_cfg['gnn_params']['gnn'].update(config)
            self._get_gnn()
        
        logger.info("Updated Hyperparameter:")
        logger.info(self.reid_net_cfg)

    def _setup_training(self):
        self.best_mota = 0
        self.best_idf1 = 0
        
        if self.reid_net_cfg['mode'] == 'hyper_search':
            self._sample_params()

        # initialize save directories
        self.save_dir = self.reid_net_cfg['save_folder_nets'] + '_inter'
        self.save_dir_final = self.reid_net_cfg['save_folder_nets']

        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        if not os.path.isdir(self.save_dir_final):
            os.mkdir(self.save_dir_final)
        
        # Save name (still to determine)
        self.fn = "_".join([self.net_type, 
                        str(self.reid_net_cfg['train_params']['lr']),
                        str(self.reid_net_cfg['train_params']['weight_decay']),
                        str(self.reid_net_cfg['train_params']['temperatur'])])
        
        # get parameters for optimizer
        if self.gnn:
            params = list(set(self.encoder.parameters())) + list(set(self.gnn.parameters()))
        else:
            params = list(set(self.encoder.parameters()))

        param_groups = [{'params': params,
                            'lr': self.reid_net_cfg['train_params']['lr']}]

        self.optimizer = ReID.RAdam.RAdam(param_groups,
                            weight_decay=self.reid_net_cfg['train_params']['weight_decay']) 
        
        # initialize losses
        self._get_loss_fns()

    def _get_loss_fns(self, ce=True, trip=False, mult_pos_contr=False, person_loss=False):
        if ce:
            self.loss1 = ReID.utils.losses.CrossEntropyLabelSmooth(
                            num_classes=self.num_classes, dev=self.device).to(self.device) 
        if trip:
            self.loss2 = ReID.utils.losses.TripletLoss() # inputs, targets --> loss, prec
        
        if mult_pos_contr:
            self.loss3 = ReID.utils.losses.MultiPositiveContrastive() # inputs, targets --> loss        
        
        if person_loss:
            self.loss4 = torch.nn.BCELoss()
    
    def _check_best(self, mota_overall, idf1_overall):
        # check if current mota better than previous
        if mota_overall > self.best_mota:
            # if so: save current checkpoints
            self.best_mota = mota_overall
            self.best_idf1 = idf1_overall
            torch.save(self.encoder.state_dict(),
                        osp.join(self.save_dir,
                                self.fn + '.pth'))
            if self.gnn:
                torch.save(self.gnn.state_dict(),
                            osp.join(self.save_dir,
                                    'gnn_' + self.fn + '.pth'))

    def _save_best(self):
        # mv best model from intermediate to final dir
        os.rename(osp.join(self.save_dir, self.fn + '.pth'),
            osp.join(self.save_dir_final, 
                    str(self.best_mota) + 
                    self.net_type + '_' + 
                    self.dataset_cfg['splits'] + '.pth'))
        if self.gnn:
            os.rename(osp.join(self.save_dir, 'gnn_' + self.fn + '.pth'),
                osp.join(self.save_dir_final, 
                        str(self.best_mota) + 'gnn_' + 
                        self.net_type + '_' +
                        self.dataset_cfg['splits'] + '.pth'))

    def _reduce_lr(self):
        logger.info("reduce learning rate")
        self.encoder.load_state_dict(torch.load(
            osp.join(self.save_dir, self.fn + '.pth')))
        self.gnn.load_state_dict(torch.load(osp.join(self.save_dir,
            'gnn_' + self.fn + '.pth')))
        for g in self.optimizer.param_groups:
            g['lr'] = self.reid_net_cfg['train_params']['lr'] / 10.
    
    def _get_summary(self, accs, names, name):
        logger.info(name)
        mh = mm.metrics.create()
        metrics = mm.metrics.motchallenge_metrics + ['num_objects', 'idtp', 
                        'idfn', 'idfp', 'num_predictions']
        summary = mh.compute_many(accs, names=names,
                                metrics=metrics,
                                generate_overall=True)
        logger.info(mm.io.render_summary(summary, formatters=mh.formatters, 
                                    namemap=mm.io.motchallenge_metric_names))

    def _get_results(self, names, corresponding_gt=None, get_gt_files=False):
        # set default solver for evaluation
        mm.lap.default_solver = 'lapsolver'

        # normal GT files or reduced gt files
        if corresponding_gt:
            gt = corresponding_gt
        else:
            gt_path = osp.join(self.dataset_cfg['mot_dir'], self.dir)
            gtfiles = [os.path.join(gt_path, i, 'gt/gt.txt') for i in names]
            gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', 
                                min_confidence=1)) for f in gtfiles])

        # GT files or normal out files
        if get_gt_files:
            out_mot_files_path = 'gt_out'
        else:
            out_mot_files_path = os.path.join('out', self.tracker.experiment)

        tsfiles = [os.path.join(out_mot_files_path, '%s' % i) for i in names]
        ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, 
                            fmt='mot15-2D')) for f in tsfiles])

        accs, names = self._compare_dataframes(gt, ts)

        return accs, names

    @staticmethod
    def _compare_dataframes(gts, ts):
        """Builds accumulator for each sequence."""
        accs = []
        names = []
        for k, tsacc in ts.items():
            if k in gts:
                accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 
                                    'iou', distth=0.5))
                names.append(k)

        return accs, names

