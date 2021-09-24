import ReID
import os.path as osp
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
#sys.path.insert(0, "/usr/wiss/seidensc/Documents/PI-ReID")
#from modeling import PISNet
#from .tracker_pi_reid import TrackerQG
sys.path.insert(0, "/usr/wiss/seidensc/Documents/batch-dropblock-network")
from get_model import get_bdb
sys.path.insert(0, "/usr/wiss/seidensc/Documents/reid-strong-baseline")
from get_model_bot import get_bot
from torchreid import models
import torchreid


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
        self._get_models()
        
        self.loaders = self._get_loaders(dataset_cfg)

        if 'train' in self.loaders.keys():
            self.num_classes = self.loaders['train'].dataset.id
            self.num_iters = 30 if self.reid_net_cfg['mode'] == 'hyper_search' else 1
        else:
            self.num_classes = self.reid_net_cfg['trained_on']['num_classes']
        
        # name of eval criteria for logging
        self.eval1 = 'MOTA'
        self.eval2 = 'IDF1'

    def _get_models(self):
        if self.reid_net_cfg['encoder_params']['net_type'] == 'query_guided':
            tracker = TrackerQG
        else:
            tracker = Tracker
        test = 1 if self.dataset_cfg['splits'] == 'mot17_test' else 0
        self._get_encoder()
        self._get_sr_gan()
        #self.proxy_gen = ProxyGenRNN(self.sz_embed).to(self.device)
        self.proxy_gen = None
        if self.reid_net_cfg['gnn']:
            self._get_gnn()
            self.tracker = tracker(self.tracker_cfg, self.encoder, 
                                        self.gnn, self.graph_gen, 
                                        self.proxy_gen, dev=self.device,
                                        net_type=self.reid_net_cfg['encoder_params']['net_type'],
                                        test=test, sr_gan=self.sr_gan)
            self.tracker.num_el_id = self.reid_net_cfg['dl_params']['num_elements_class']
        else:
            self.gnn, self.graph_gen = None, None
            self.tracker = tracker(self.tracker_cfg, self.encoder, 
                                    proxy_gen=self.proxy_gen, dev=self.device, 
                                    net_type=self.reid_net_cfg['encoder_params']['net_type'],
                                    test=test, sr_gan=self.sr_gan)

    def _get_sr_gan(self):
        if self.tracker_cfg["use_sr_gan"]:
            args = self._get_gan_args()
            self.sr_gan = configure(args)
            self.sr_gan = self.sr_gan.to(self.device)
        else:
            self.sr_gan = None
    
    def _get_gan_args(self):
        parser = argparse.ArgumentParser("Photo-Realistic Single Image Super-Resolution Using "
                                 "a Generative Adversarial Network.")
        parser.add_argument("-a", "--arch", metavar="ARCH", default="srgan")
        parser.add_argument("--upscale-factor", type=int, default=4, choices=[2, 4, 8],
                            help="Low to high resolution scaling factor. Optional: [2, 4, 8] (default: 4)")
        parser.add_argument("--model-path", default="../SRGAN-PyTorch/weights/up4_64_32.pth", type=str, metavar="PATH",
                            help="Path to latest checkpoint for model.")
        parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                            help="Use pre-trained model.")
        parser.add_argument("--seed", default=None, type=int,
                            help="Seed for initializing training.")
        parser.add_argument("--gpu", default=None, type=int,
                            help="GPU id to use.")

        return parser.parse_args()

    def _get_encoder(self):
        if self.reid_net_cfg['encoder_params']['net_type'] == 'query_guided':
            encoder = PISNet(483, 1, '/usr/wiss/seidensc/Documents/PI-ReID/pretrained/prw/resnet50_model_120.pth', 'bnneck', 'after', 'resnet50', 'self', has_non_local='yes', sia_reg='yes', pyramid='no', test_pair='no')
            pretrained_dic = torch.load('/usr/wiss/seidensc/Documents/PI-ReID/output/prw/resnet50_checkpoint_1280.pt')['model']#.state_dict()
            model_dict = encoder.state_dict()
            model_dict.update(pretrained_dic)
            encoder.load_state_dict(model_dict)
            self.sz_embed = None
            self.encoder = encoder.to(self.device)
        elif self.reid_net_cfg['encoder_params']['net_type'] == 'batch_drop_block':
            encoder = get_bdb()
            self.encoder = encoder.to(self.device)
            self.sz_embed = None
        elif self.reid_net_cfg['encoder_params']['net_type'] == 'bot':
            encoder = get_bot()
            self.encoder = encoder.to(self.device)
            self.sz_embed = None
        elif self.reid_net_cfg['encoder_params']['net_type'] == 'resnet50_analysis':
            encoder = models.build_model(name='resnet50', num_classes=1000)
            torchreid.utils.load_pretrained_weights(encoder, 'resnet50_market_xent.pth.tar') 
            self.sz_embed = None
        else:
            encoder, self.sz_embed = ReID.net.load_net(
                self.reid_net_cfg['trained_on']['name'],
                self.num_classes, 'test',
                attention=False,
                **self.reid_net_cfg['encoder_params'])

        self.encoder = encoder.to(self.device)

        def get_n_params(model):
            pp=0
            for p in list(model.parameters()):
                nn=1
                for s in list(p.size()):
                    nn = nn*s
                pp += nn
            return pp
        print("Number of parameters of encoder {}".format(get_n_params(self.encoder)))
    
    def _get_gnn(self):
        self.reid_net_cfg['gnn_params']['classifier']['num_classes'] = self.num_classes
        if self.reid_net_cfg['encoder_params']['net_type'] == 'resnet50FPN':
            self.gnn = ReID.net.GNNReID(self.device,
                                self.reid_net_cfg['gnn_params'],
                                4 * self.sz_embed).to(self.device)
        elif self.reid_net_cfg['encoder_params']['net_type'] == 'resnet50_attention':
                self.gnn = ReID.net.SpatialGNNReID(self.device, 
                                self.reid_net_cfg['gnn_params']).to(self.device)
        else:
            self.gnn = ReID.net.GNNReID(self.device,
                                self.reid_net_cfg['gnn_params'],
                                self.sz_embed).to(self.device)

        if self.reid_net_cfg['gnn_params']['pretrained_path'] != "no":
            load_dict = torch.load(
                self.reid_net_cfg['gnn_params']['pretrained_path'],
                map_location='cpu')

            load_dict = {k: v for k, v in load_dict.items() if 'fc' 
                            not in k.split('.')}
            '''
            ld = dict()
            for k, v in load_dict.items():
                if k[:3] == 'gnn' or k[:3] == 'bot' or k[:2] == 'fc':
                    if '0' in k.split('.'):
                        ld[k] = v
                else:
                    ld[k] = v
            '''
            ld = load_dict
            
            gnn_dict = self.gnn.state_dict()
            gnn_dict.update(ld) #(load_dict)
                
            self.gnn.load_state_dict(gnn_dict)

        self.graph_gen = ReID.net.GraphGenerator(self.device,
                                                    **self.reid_net_cfg[
                                                      'graph_params'])
        #self.tracker = Tracker(self.tracker_cfg, self.encoder, self.gnn, 
        #                        self.graph_gen)
    
    def _get_loaders(self, dataset_cfg):
        loaders = dict()
        for mode in _SPLITS[dataset_cfg['splits']].keys():
            seqs = _SPLITS[dataset_cfg['splits']][mode]['seq']
            self.dir = _SPLITS[dataset_cfg['splits']][mode]['dir']
            if mode != 'train':
                dataset = TrackingDataset(dataset_cfg['splits'], seqs, 
                                            dataset_cfg, self.dir, dev=self.device)
                loaders[mode] = DataLoader(dataset, batch_size=1, shuffle=False,
                                        collate_fn=collate_test)
            else:
                dataset = MOTDataset(dataset_cfg['splits'], seqs, 
                                        dataset_cfg, self.dir) 
                loaders[mode] = DataLoader(dataset, batch_size=1, shuffle=False,
                                        collate_fn=collate_train)
        return loaders
    
    def _sample_params(self):

        config = {'lr': 10 ** random.uniform(-5, -3),
                  'weight_decay': 10 ** random.uniform(-15, -6),
                  'temperatur': random.random(),
                  'epochs': 5}
        self.reid_net_cfg['train_params'].update(config)
        
        if self.reid_net_cfg['gnn_params']['pretrained_path'] == 'no':
            config = {'num_layers': random.randint(1, 4)}
            config = {'num_heads': random.choice([1, 2, 4, 8])}

            self.reid_net_cfg['gnn_params']['gnn'].update(config)
            self._get_gnn()
        
        logger.info("Updated Hyperparameter:")
        logger.info(self.reid_net_cfg)

    def _setup_training(self):
        self.best_mota = 0
        self.best_idf1 = 0
        
        if self.reid_net_cfg['mode'] == 'hyper_search':
            self._sample_params()

        self.save_folder_nets = self.reid_net_cfg['save_folder_nets'] + '_inter'
        self.save_folder_nets_final = self.reid_net_cfg['save_folder_nets']
        if not os.path.isdir(self.save_folder_nets):
            os.mkdir(self.save_folder_nets)
        if not os.path.isdir(self.save_folder_nets_final):
            os.mkdir(self.save_folder_nets_final)
        self.fn = str(time.time())
        
        if self.gnn:
            params = list(set(self.encoder.parameters())) + list(set(self.gnn.parameters()))
        else:
            params = list(set(self.encoder.parameters()))

        param_groups = [{'params': params,
                            'lr': self.reid_net_cfg['train_params']['lr']}]

        self.optimizer = ReID.RAdam.RAdam(param_groups,
                            weight_decay=self.reid_net_cfg['train_params']['weight_decay']) 
        
        self.loss1 = ReID.utils.losses.CrossEntropyLabelSmooth(
                        num_classes=self.num_classes, dev=self.device).to(self.device) 
        self.loss2 = ReID.utils.losses.CrossEntropyLabelSmooth(
                        num_classes=self.num_classes, dev=self.device).to(self.device)
        
    
    def _check_best(self, mota_overall, idf1_overall):
        if mota_overall > self.best_mota:
                self.best_mota = mota_overall
                self.best_idf1 = idf1_overall
                torch.save(self.encoder.state_dict(),
                            osp.join(self.save_folder_nets,
                                    self.fn + '.pth'))
                torch.save(self.gnn.state_dict(),
                            osp.join(self.save_folder_nets,
                                    'gnn_' + self.fn + '.pth'))

    def _save_best(self):
        os.rename(osp.join(self.save_folder_nets, self.fn + '.pth'),
            osp.join(self.save_folder_nets_final, 
                    str(self.best_mota) + 
                    self.reid_net_cfg['encoder_params']['net_type'] + '_' + 
                    self.dataset_cfg['splits'] + '.pth'))
        os.rename(osp.join(self.save_folder_nets, 'gnn_' + self.fn + '.pth'),
            osp.join(self.save_folder_nets_final, 
                    str(self.best_mota) + 'gnn_' + 
                    self.reid_net_cfg['encoder_params']['net_type'] + '_' +
                    self.dataset_cfg['splits'] + '.pth'))

    def _reduce_lr(self):
        logger.info("reduce learning rate")
        self.encoder.load_state_dict(torch.load(
            osp.join(self.save_folder_nets, self.fn + '.pth')))
        self.gnn.load_state_dict(torch.load(osp.join(self.save_folder_nets,
            'gnn_' + self.fn + '.pth')))
        for g in self.optimizer.param_groups:
            g['lr'] = self.reid_net_cfg['train_params']['lr'] / 10.

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
            if e == 31 or e == 51:
                self._reduce_lr()
            self.optimizer.zero_grad()

            if type(data[0]) == list: # data == graph of n frames
                img, Y = data[0][0], data[1][0]
            else: # data == bboxes of persons (reid setting)
                img, Y = data[0], data[1]

            Y, img = Y.to(self.device), img.to(self.device)
            preds1, feats1 = self.encoder(img, output_option='plain')
            
            loss = self.loss1(preds1/temp, Y)
            losses['Loss 1'].append(loss)
            
            if self.gnn:
                edge_attr, edge_ind, feats1 = self.graph_gen.get_graph(feats1)
                preds2, feats2 = self.gnn(feats1, edge_ind, edge_attr, 'plain')
                loss2 = self.loss2(preds2[-1]/temp, Y)
                losses['Loss 2'].append(loss2)

                loss += loss2
                losses['Total'].append(loss)

            loss.backward()
            self.optimizer.step()

        logger.info({k: sum(v)/len(v) for k, v in losses.items()})     

    def _double_check(self, seq):
        import pandas as pd
        d = osp.join('../../datasets/MOT/MOT17Labels/train/', seq[0].name, 'det', 'tracktor_prepr_det.txt')
        dets = pd.read_csv(d, names = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'label', 'vis', '?'])
        m = os.path.join('out', self.tracker.experiment, seq[0].name)
        mine = pd.read_csv(m, names = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'label', 'vis', '?'])
        for i in range(1, 1+max(dets['frame'].unique())):
            m_f = mine[mine['frame'] == i]
            d_f = dets[dets['frame'] == i]

            def check(m_f, d_f, name, direction, frame):
                for s1 in set(m_f[name]):
                    c = 0
                    for s2 in set(d_f[name]):
                        if abs(s1-s2) < 0.0001:
                            c += 1
                    if c == 0:
                        logger.info("{}, {},  {}, {}, {}".format(frame, direction, name, s1, set(d_f[name])))
            
            check(m_f, d_f, 'bb_left', "md", i)
            check(m_f, d_f, 'bb_top', "md", i)
            check(m_f, d_f, 'bb_width', "md", i)
            check(m_f, d_f, 'bb_height', "md", i)
            check(d_f, m_f, 'bb_left', "dm", i)
            check(d_f, m_f, 'bb_top', "dm", i)
            check(d_f, m_f, 'bb_width', "dm", i)
            check(d_f, m_f, 'bb_height', "dm", i)         
            #print(dets[dets['frame'] == i].shape[0], mine[mine['frame'] == i].shape[0])                                                   

    def _evaluate(self, mode='val'):
        names = list()
        corresponding_gt = OrderedDict()
        counter_iou = 0
        for seq in self.loaders[mode]:
            #if self.tracker_cfg['oracle']:
            if self.dataset_cfg['splits'] != 'mot17_test':
                df = seq[0].corresponding_gt
                df = df.drop(['bb_bot', 'bb_right'], axis=1)
                df = df.rename(columns={"frame": "FrameId", "id": "Id", "bb_left": "X", "bb_top": "Y", "bb_width": "Width", "bb_height": "Height", "conf": "Confidence", "label": "ClassId", "vis": "Visibility"})
                df = df.set_index(['FrameId', 'Id'])
                corresponding_gt[seq[0].name] = df
            self.tracker.gnn, self.tracker.encoder = self.gnn, self.encoder
            self.tracker.track(seq[0])
            names.append(seq[0].name)
            #self._double_check(seq)
        counter_iou = self.tracker.counter_iou
        logger.info(counter_iou)
        if self.dataset_cfg['splits'] != 'mot17_test':
            if self.tracker_cfg['oracle']:
                # Oracle GT
                accs, names = self._get_results(names, corresponding_gt)
                self._get_summary(accs, names, "Oracle GT")

                if self.dataset_cfg['save_oracle']:
                    # Oracle Output
                    accs, names = self._get_results(names, get_gt_files=True)
                    self._get_summary(accs, names, "Oracle results")

            # Normal 
            accs, names = self._get_results(names)
            self._get_summary(accs, names, "Normal")

        return None, None 
    
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

