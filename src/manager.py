import ReID
from ReID import net
import os.path as osp

import os
from data.splits import _SPLITS
from src.datasets.TrainDatasetWeightPred import TrainDatasetWeightPred
from src.tracking_utils import TripletLoss, WeightPredictor, collate, get_precision
from .tracker import Tracker
from src.datasets.TrackingDataset import TrackingDataset
import logging
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path
from torchreid import models
import torchreid
from src.eval_fairmot import Evaluator
from src.eval_mpn_track import get_results, get_summary
from src.eval_track_eval import evaluate_track_eval
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


frames = {'JDE': [299, 524, 418, 262, 326, 449, 368],
          'CSTrack': [298, 523, 417, 261, 325, 448, 368],
          'TraDeS': [299, 524, 418, 262, 326, 449, 374],
          'CenterTrack': [299, 524, 418, 262, 326, 449, 374],
          'CenterTrackPub': [299, 524, 418, 262, 326, 449, 374],
          'qdtrack': [299, 524, 418, 262, 326, 449, 374],
          'FairMOT': [300, 525, 419, 263, 327, 450, 369],
          'TransTrack': [299, 524, 418, 262, 326, 449, 374],
          'center_track': [300, 525, 419, 263, 327, 450, 375],
          'tracktor_prepr_det': [300, 525, 419, 263, 327, 450, 375]}


logger = logging.getLogger('AllReIDTracker.Manager')
# logger.propagate = False


class Manager():
    def __init__(self, device, dataset_cfg, reid_net_cfg, tracker_cfg, cfg):
        self.device = device
        self.reid_net_cfg = reid_net_cfg
        self.dataset_cfg = dataset_cfg
        self.tracker_cfg = tracker_cfg
        self.cfg = cfg

        # load ReID net
        self.loaders = self._get_loaders(dataset_cfg)
        self._get_models()

    def _train(self, train_cfg):
        if train_cfg['loss'] == 'triplet':
            criterion = TripletLoss(train_cfg['margin'])
        elif train_cfg['loss'] == 'l2':
            criterion  = nn.MSELoss()
        elif train_cfg['loss'] == 'l2_weighted':
            criterion  = nn.MSELoss(reduce=False)
        if train_cfg['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                self.weight_pred.parameters(),
                lr=train_cfg['lr']/10,
                weight_decay=train_cfg['wd'],
                momentum=0.9)
        else:
            optimizer = optim.Adam(
                self.weight_pred.parameters(),
                lr=train_cfg['lr'],
                weight_decay=train_cfg['wd'])

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30], gamma=0.1)

        params = str(train_cfg['margin']) + \
            str(train_cfg['lr']) + \
            str(train_cfg['scale_loss']) + \
            str(train_cfg['loss']) + \
            str(train_cfg['input_dim']) + \
            str(train_cfg['wd']) + \
            str(train_cfg['optimizer'])

        self.writer = SummaryWriter('runs/train_weight_pred_triplet' + params)
        # evaluate before training
        self.weight_pred.eval()
        mota, idf1 = 0, 0
        # mota, idf1 = self._evaluate('val', log=False)
        best_idf1 = idf1
        logger.info('Before Training MOTA: {}, IDF1: {}, best IDF1 {}'.format(
                mota, idf1, best_idf1))

        if train_cfg['load'] != 'no':
            chckpt = torch.load(train_cfg['load'])
            start_epoch = chckpt['epoch']
            state_dict = chckpt['model_state_dict']
            optim_dict = chckpt['optimizer_state_dict']
            last_loss = chckpt['loss']
            self.weight_pred.load_state_dict(state_dict)
            optimizer.load_state_dict(optim_dict)
            logger.info('Load model from {}, last loss {}, epoch {}'.format(
                train_cfg['load'],
                last_loss,
                start_epoch
            ))
        else:
            start_epoch = 0

        for e in range(start_epoch, train_cfg['num_epochs']):
            logger.info('Starting epoch {}/{}...'.format(e, train_cfg['num_epochs']))
            # train epoch 
            self.weight_pred.train()
            loss_list = list()

            for data in self.loaders['train']:
                w1s, w2s, w3s, scs, ioas, embs = data
                optimizer.zero_grad()
                loss = 0
                precs = 0
                for w1, w2, w3, sc, ioa, emb in zip(w1s, w2s, w3s, scs, ioas, embs):
                    # zero grad optimizer
                    
                    # get weights
                    w1 = torch.flatten(w1)
                    w2 = torch.flatten(w2)
                    w3 = torch.flatten(w3)
                    if train_cfg['input_dim'] == 3:
                        inp = torch.stack([w1, w2, w3]).to(self.device).T
                        weights = self.weight_pred(inp)
                        weights_emb = weights.reshape(emb.shape)
                        weights_ioa = (1 - weights).reshape(emb.shape)

                        # make dist with weights
                        dist = emb.to(self.device) * weights_emb + \
                            ioa.to(self.device) * weights_ioa
                    else:
                        ioa = torch.flatten(ioa)
                        emb = torch.flatten(emb)
                        inp = torch.stack([w1, w2, w3, emb, ioa]).to(
                            self.device).T
                        dist = self.weight_pred(inp).reshape(sc.shape)

                    # compute loss and backprop
                    if train_cfg['loss'] == 'triplet':
                        loss, prec = criterion(dist, sc.to(self.device))
                    elif train_cfg['loss'] == 'l2':
                        target = ~sc
                        loss = criterion(torch.flatten(dist), torch.flatten(
                            target.to(self.device)).float())
                        prec = get_precision(dist.reshape(sc.shape), sc.to(
                            self.device))
                    elif train_cfg['loss'] == 'l2_weighted':
                        # compute l2 loss
                        target = ~sc
                        loss = criterion(torch.flatten(dist), torch.flatten(
                            target.to(self.device)).float())

                        # multiply loss with weight
                        weight = torch.ones(loss.shape)
                        weight[torch.flatten(sc)] *= train_cfg['weight']
                        weight = weight.to(self.device)
                        loss = (loss * weight).mean()

                        # get precision
                        prec = get_precision(dist.reshape(sc.shape), sc.to(
                            self.device))

                    loss += loss
                    precs += prec
                loss = loss/len(embs)
                precs = precs/len(embs)
                loss = loss * train_cfg['scale_loss']
                loss.backward()
                optimizer.step()

                self.writer.add_scalar('Loss/train', loss.item(), e)
                self.writer.add_scalar('Loss/precision', precs.item(), e)
                loss_list.append([loss.item(), precs.item()])

            avg_loss = sum([l[0] for l in loss_list])/len(loss_list)
            avg_prec = sum([l[1] for l in loss_list])/len(loss_list)
            logger.info('Average loss: {}, average precision: {}'.format(
                avg_loss, avg_prec))

            scheduler.step()

            # evaluate epoch
            logger.info('Evaluating...')
            self.weight_pred.eval()
            mota, idf1 = self._evaluate('val', log=False)

            self.writer.add_scalar('MOTA/val', mota, e)
            self.writer.add_scalar('IDF1/val', idf1, e)

            logger.info('MOTA: {}, IDF1: {}, best IDF1 {}'.format(
                mota, idf1, best_idf1))

            if idf1 > best_idf1:
                torch.save({
                    'epoch': train_cfg['num_epochs'],
                    'model_state_dict': self.weight_pred.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, 'weight_pred/' + params + 'checkpoint.tar')
                best_idf1 = idf1
            
            if e in [10, 30]:
                state = torch.load('weight_pred/' + params + 'checkpoint.tar')
                self.weight_pred.load_state_dict(state['model_state_dict'])

        self.writer.close()

        os.makedirs('weight_pred', exist_ok=True)

    def _evaluate(self, mode='val', first=False, log=True):
        print(log)
        names = list()
        corresponding_gt = OrderedDict()

        # get tracking files
        print(self.loaders)
        for seq in self.loaders[mode]:
            # first = feed sequence data through backbon and update statistics
            # before tracking
            if first:
                self.reset_for_first(seq)

            # get gt bbs corresponding to detections for oracle evaluations
            if self.dataset_cfg['splits'] != 'mot17_test' and self.dataset_cfg['splits'] != 'mot20_test':
                self.get_corresponding_gt(seq, corresponding_gt)

            self.tracker.encoder = self.encoder
            self.tracker.track(seq[0], log=log)
            names.append(seq[0].name)

        # print interaction and occlusion stats
        self.print_interaction_occlusion_stats()

        # manually set experiment if already generated bbs
        # self.tracker.experiment = osp.join(
        #     'OtherTrackersOrig',
        #     # 'OtherTrackersOrigMOT20',
        #     self.dataset_cfg['det_file'][:-4])
        if log:
            logger.info(self.tracker.experiment)

        # EVALUATION FAIRMOT
        self.eval_fair_mot(names)

        # EVALUATION FROM MPNTRACK
        # self.eval_mpn_track(names, corresponding_gt)

        # EVALUATION TRACKEVAL
        output_res, _ = self.eval_track_eval(log)
        mota = output_res['MotChallenge2DBox'][self.tracker.experiment][
            'COMBINED_SEQ']['pedestrian']['CLEAR']['MOTA']
        idf1 = output_res['MotChallenge2DBox'][self.tracker.experiment][
            'COMBINED_SEQ']['pedestrian']['Identity']['IDF1']

        return mota, idf1

    def _get_models(self):
        if self.tracker_cfg['motion_config']['ioa_threshold'] == 'learned':
            self.weight_pred = WeightPredictor(self.cfg['train']['input_dim'], 1).to(self.device)
            self.weight_pred.eval()
        else:
            self.weight_pred = None

        self._get_encoder()
        weight = self.reid_net_cfg['encoder_params']['pretrained_path'].split(
            '/')[-1][:8] if self.reid_net_cfg['encoder_params']['net_type'][
                :8] == 'resnet50' and self.reid_net_cfg['encoder_params'][
                    'net_type'][:8] != 'resnet50_analysis' else ''
        self.tracker = Tracker(self.tracker_cfg, self.encoder,
                               net_type=self.net_type,
                               output=self.reid_net_cfg['output'],
                               weight=weight,
                               data=self.dataset_cfg['det_file'],
                               weight_pred=self.weight_pred,
                               device=self.device,
                               train_cfg=self.cfg['train'])

    def _get_encoder(self):
        self.net_type = self.reid_net_cfg['encoder_params']['net_type']
        if self.net_type == 'resnet50_analysis':
            # get pretrained resnet 50 from torchreid
            encoder = torchreid.models.build_model(
                name='resnet50', num_classes=1000)
            torchreid.utils.load_pretrained_weights(
                encoder, 'resnet50_market_xent.pth.tar')
            self.sz_embed = None

        else:
            # own trained network
            encoder, self.sz_embed = net.load_net(
                self.reid_net_cfg['trained_on']['name'],
                self.reid_net_cfg['trained_on']['num_classes'],
                'test', attention=False,
                **self.reid_net_cfg['encoder_params'])

        self.encoder = encoder.to(self.device)

    def _get_loaders(self, dataset_cfg):
        # Initialize datasets
        loaders = dict()
        # get loaders for test / val mode of current split
        for mode in _SPLITS[dataset_cfg['splits']].keys():
            seqs = _SPLITS[dataset_cfg['splits']][mode]['seq']
            self.dir = _SPLITS[dataset_cfg['splits']][mode]['dir']

            # tracking dataset
            if mode != 'train':
                dataset = TrackingDataset(
                    dataset_cfg['splits'],
                    seqs,
                    dataset_cfg,
                    self.dir,
                    dev=self.device)
                loaders[mode] = dataset
            elif mode == 'train':
                dataset = TrainDatasetWeightPred(
                    path=self.cfg['train']['path']
                )
                dl = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.cfg['train']['bs'],
                    shuffle=True,
                    sampler=None,
                    collate_fn=collate,
                    num_workers=0)

                loaders[mode] = dl



        return loaders

    def reset_for_first(self, seq):
        experiment = self.tracker.experiment
        self._get_models()
        self.tracker.experiment = experiment
        self.tracker.encoder = self.encoder
        self.tracker.track(seq[0], first=True)

    def get_corresponding_gt(self, seq, corresponding_gt):
        df = seq[0].corresponding_gt
        df = df.drop(['bb_bot', 'bb_right'], axis=1)
        df = df.rename(
            columns={
                "frame": "FrameId",
                "id": "Id",
                "bb_left": "X",
                "bb_top": "Y",
                "bb_width": "Width",
                "bb_height": "Height",
                "conf": "Confidence",
                "label": "ClassId",
                "vis": "Visibility"})
        df = df.set_index(['FrameId', 'Id'])
        corresponding_gt[seq[0].name] = df

    def print_interaction_occlusion_stats(self):
        for what, dd in zip(['Interaction', 'Occlusion'], [
                            self.tracker.interaction, self.tracker.occlusion]):
            logger.info('{} statistics'.format(what))
            for k, v in dd.items():
                logger.info(
                    '{}: \t {} \t {}'.format(
                        k, sum(v) / len(v), len(v)))
                print('{}: \t {} \t {}'.format(
                        k, sum(v) / len(v), len(v)))

    def eval_fair_mot(self, names):
        # EVALUATION FROM FAIRMOT oder so
        accs = list()
        for seq in names:
            evaluator = Evaluator(
                '/storage/slurm/seidensc/datasets/MOT/MOT17/train', seq, 'mot')
            accs.append(
                evaluator.eval_file(
                    os.path.join(
                        'out/' +
                        self.tracker.experiment,
                        seq)))

        metrics = mm.metrics.motchallenge_metrics
        mh = mm.metrics.create()
        summary = Evaluator.get_summary(accs, names, metrics)
        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )
        logger.info(strsummary)

    def eval_mpn_track(self, names, corresponding_gt):
        if self.dataset_cfg['splits'] != 'mot17_test':
            if self.tracker_cfg['oracle']:
                # Use corresponding gt as gt
                # --> no false negatives
                accs, names = get_results(
                    names,
                    corr_gt_as_gt=corresponding_gt,
                    dir=self.dir,
                    tracker=self.tracker_cfg,
                    dataset_cfg=self.dataset_cfg)
                get_summary(accs, names, "Corresponding GT as GT")

                # Use corresponding gt as tracking results
                # --> perfect association, only FNs
                # --> how good can we be if we associate correctly
                accs, names = get_results(
                    names,
                    corr_gt_as_track_res=corresponding_gt,
                    dir=self.dir,
                    tracker=self.tracker_cfg,
                    dataset_cfg=self.dataset_cfg)
                get_summary(accs, names, "Corresponding GT as results")

            # Evaluate files from tracking
            accs, names = get_results(
                names,
                dir=self.dir,
                tracker=self.tracker,
                dataset_cfg=self.dataset_cfg)
            get_summary(accs, names, "Normal")

    def eval_track_eval(self, log=True):
        output_res, output_msg = evaluate_track_eval(
            dir=self.dir,
            tracker=self.tracker,
            dataset_cfg=self.dataset_cfg,
            log=log
        )

        return output_res, output_msg
