from . import calc_mean_average_precision
import torch
import logging
import json
from collections import defaultdict
import sklearn.cluster
import sklearn.metrics.cluster
import os
from sklearn.manifold import TSNE
import numpy as np
import cv2

logger = logging.getLogger('GNNReID.Evaluator')

class Evaluator():
    def __init__(self, lamb, k1, k2, output_test_enc='norm', output_test_gnn='norm', re_rank=False, cat=0):
        self.lamb = lamb
        self.k1 = k1
        self.k2 = k2
        self.output_test = output_test_enc
        self.re_rank = re_rank

    def evaluate(self, model, dataloader, query=None, gallery=None, 
            gnn=None, graph_generator=None, add_dist=False, batchwise_rank=False,
            query_guided=False, dl_ev_gnn=None, queryguided=False, 
            attention=False):
        # query_guided == use QG network
        # queryguided == evaluate for each gallery and each query
        model_is_training = model.training
        model.eval()

        # use backbone and gnn for evaluation with sampled batches
        # and evaluate in every batch
        if batchwise_rank:
            logger.info('evaluate using backbone + gnn per batch')
            gnn_is_training = gnn.training
            gnn.eval()

            mAP, cmc = self.predict_reid_features_gnn(model, gnn,
                    graph_generator, dataloader, batchwise_rank=True,
                    query_guided=query_guided, attention=attention)

            gnn.train(gnn_is_training)
        
        # query guided evaluation --> for each query compute diff gallery feats
        elif queryguided:
            logger.info('evaluate using QG setting')
            gnn_is_training = gnn.training
            gnn.eval()

            mAP, cmc = self.predict_reid_features_qg(model, gnn,
                    dataloader, dataloader_queryguided=dl_ev_gnn, 
                    query=query, gallery=gallery)

            gnn.train(gnn_is_training)
        else:
            # just use backbone for evaluation
            if not gnn: 
                logger.info('evaluate using only backbone')
                features, labels, camids = self.predict_reid_features_bb(model, \
                    dataloader, add_dist, attention)

            # use backbone and gnn for evaluation with sampled batches
            else: 
                logger.info('evaluate using backbone + gnn using sampled batches')
                gnn_is_training = gnn.training
                gnn.eval()

                features, labels, camids = self.predict_reid_features_gnn(model, gnn,
                        graph_generator, dataloader, attention=attention)

                gnn.train(gnn_is_training)
            
            if len(camids):
                logger.info("Using MOT17 - use camids and labels from dataset directly")
                qc = np.array([camids[k] for k in query])
                gc = np.array([camids[k] for k in gallery])
                qi = np.array([labels[k] for k in query])
                gi = np.array([labels[k] for k in gallery])

            else:
                qc = gc = qi = gi = None

            mAP, cmc = calc_mean_average_precision(features, query, gallery,
                                                gc=gc, qc=qc, gi=gi, qi=qi) 
        model.train(model_is_training)
        return mAP, cmc

    def predict_reid_features_bb(self, model, dataloader, add_dist, attention=False):
        # just use the backbone to predict reid features
        # camids and labels only needed for MOT17 evaluation
        MOT17 = False
        features, labels, camids = dict(), dict(), dict()

        with torch.no_grad():
            for X, Y, I, P in dataloader:
                # separate camids and path for MOT17
                if len(P) == 2:
                    MOT17 = True
                    camid = P[1]
                    P = P[0]

                if torch.cuda.is_available(): X = X.cuda()

                if add_dist or attention:
                    _, fc7, _ = model(X, output_option=self.output_test, val=True)
                else:
                    _, fc7 = model(X, output_option=self.output_test, val=True)

                # this accounts for FPN features (4 features per samp)
                if type(fc7) == list:
                    fc7 = torch.cat(fc7, dim=1)
                
                # add features to dict, labels and camid only for MOT17
                for path, out in zip(P, fc7):
                    features[path] = out.detach()
                if MOT17:
                    for path, c, y in zip(P, camid, Y):
                        camids[path] = c
                        labels[path] = y.detach()

        return features, labels, camids

    def predict_reid_features_gnn(self, model, gnn, graph_generator, \
            dataloader, attention=True, batchwise_rank=False, query_guided=False):
        MOT17 = False
        # apply a 
        features = dict()
        labels = dict()
        camids = dict()

        CMC, mAP = list(), list()
        with torch.no_grad():
            for X, Y, I, P in dataloader:
                if len(P) == 2:
                    MOT17 = True
                    camid = P[1]
                    P = P[0]

                if torch.cuda.is_available(): X = X.cuda()

                # get backbone features
                if attention:
                    _, queries, fc7 = model(X, output_option=self.output_test, val=True)
                else:
                    _, fc7 = model(X, output_option=self.output_test, val=True) 
                
                if not query_guided:
                    edge_attr, edge_index, fc7 = graph_generator.get_graph(fc7)
                    _, fc7 = gnn(fc7, edge_index, edge_attr,
                                    output_option=self.output_test)
                
                else:
                    _, _, _, qs, gs, attended_feats, fc_x = gnn(fc7)
                    attended_feats = attended_feats.reshape(X.shape[0], X.shape[0]-1, -1)

                    dist = [sklearn.metrics.pairwise_distances(q.unsqueeze(0).cpu().numpy(), \
                        f.cpu().numpy(), metric='euclidean') for q, f in zip(queries, attended_feats)]
                    dist = np.asarray([np.insert(d, i, 0) for i, d in enumerate(dist)])

                for path, out, y in zip(P, fc7[-1], Y):
                    features[path] = out.detach()

                if MOT17:
                    for path, c in zip(P, camid):
                        camids[path] = c
                        labels[path] = y.detach()
                    gc=camids[1:], qc=[camid[0]], gi=labels[1:], qi=[labels[0]]
                else:
                    gc = qc = gi = qi = None
                
                if batchwise_rank or query_guided:
                    if not query_guided:
                        qg = list(features.keys())
                        maP, cmc = calc_mean_average_precision(features.detach(), qg, qg, \
                            gc=gc, qc=qc, gi=gi, qi=qi)
                    else:
                        maP, cmc = calc_mean_average_precision(None, P, P, distmat=dist, \
                            gc=gc, qc=qc, gi=gi, qi=qi)

                    CMC.append(cmc)
                    mAP.append(maP)

                    features, labels, camids = dict(), dict(), dict()

        if batchwise_rank or query_guided:
            cmc_avg = dict()
            for k in cmc.keys():
                eval_type = list()
                for data in CMC:
                    eval_type.append(data[k])
                eval_type = np.mean(np.stack(eval_type), axis=0)
                cmc_avg[k] = eval_type
            #logger.info(sum(mAP)/len(mAP), cmc_avg['Market'][0])
            return sum(mAP)/len(mAP), cmc_avg
        
        return features, labels, camids

    def predict_reid_features_qg(self, model, gnn, dataloader, query=None, \
            gallery=None, dataloader_queryguided=None, visualize=False):

        MOT17 = False
        features = dict()
        feature_maps = dict()
        camids = dict()
        CMC, mAP = list(), list()
        indices = dict()
        with torch.no_grad():
            for X, Y, I, P in dataloader:
                if len(P) == 2:
                    MOT17 = True
                    camid = P[1]
                    P = P[0]

                if torch.cuda.is_available(): X = X.cuda()

                _, queries, fc7 = model(X, output_option=self.output_test, val=True)

                for path, out, f, i in zip(P, queries, fc7, I):
                    features[path] = out.detach()
                    indices[path] = i
                    feature_maps[path] = f.detach()

            x = torch.cat([features[f].unsqueeze(0) for f in query], 0)
            y = torch.cat([features[f].unsqueeze(0) for f in gallery], 0)
            indices_query = [indices[f].unsqueeze(0) for f in query]
            indices_gallery = [indices[f].unsqueeze(0) for f in gallery]

            dataloader_queryguided.sampler.indices_query = indices_query
            dataloader_queryguided.sampler.indices_gallery = indices_gallery
            import time
            start = time.time()
            logger.info("Start evaluation")
            for i in range(len(indices_query)):
                if i % 100 == 0:
                    logger.info('Evaluate query {}/{}'.format(i, len(indices_query)))
                    if i == 0:
                        s = time.time()
                    else:
                        logger.info("Took {} for 100 queries".format(time.time()-s))
                        s = time.time()
                dataloader_queryguided.sampler.current_query_index = i
                dataloader_queryguided.dataset.no_imgs = True
                qg_dists, camids, labels = dict(), dict(), dict()
                for Y, I, P in dataloader_queryguided:
                    if type(P[0]) ==  tuple:
                        MOT17 = True
                        camid = [p[1] for p in P]
                        P = [p[0] for p in P]
                    
                    # get feature maps
                    fc7 = torch.cat([feature_maps[p].unsqueeze(0) for p in P], 0)

                    # get attended features
                    _, _, _, qs, gs, attended_feats, _, att_g = gnn(fc7, num_query=1)

                    if visualize:
                        visualize_att_map(att_g, P)

                    # get distance
                    dist = sklearn.metrics.pairwise_distances(features[P[0]].unsqueeze(0).cpu().numpy(), \
                         attended_feats.cpu().numpy(), metric='euclidean')

                    # with self attention
                    #dist = sklearn.metrics.pairwise_distances(attended_feats[0].unsqueeze(0).cpu().numpy(), \
                    #    attended_feats[1:].cpu().numpy(), metric='euclidean')
                    
                    for d, p in zip(dist[0], P[1:]):
                        qg_dists[p] = d

                    if MOT17:
                        for path, c, y in zip(P[1:], camid[1:], Y):
                            labels[path] = y
                            camids[path] = c
                    
                dist = np.asarray([v for v in qg_dists.values()])
                gallery = [k for k in qg_dists.keys()]
                query = [P[0]]

                if MOT17:
                    camids = [v for v in camids.values()]
                    labels = [v for v in labels.values()]
                    gc=camids[1:], qc=[camid[0]], gi=labels[1:], qi=[labels[0]]
                else:
                    gc = qc = gi = qi = None

                maP, cmc = calc_mean_average_precision(None, query=query, gallery=gallery, \
                    distmat=dist, gc=gc, qc=qc, gi=gi, qi=qi)

                if cmc is not None and maP is not None:
                    CMC.append(cmc)
                    mAP.append(maP)

        logger.info("Evaluation took {}".format(time.time()-start))
        if len(CMC) > 0:
            cmc_avg = dict()
            for k in cmc.keys():
                eval_type = list()
                for data in CMC:
                    eval_type.append(data[k])
                eval_type = np.mean(np.stack(eval_type), axis=0)
                cmc_avg[k] = eval_type
            #logger.info(sum(mAP)/len(mAP), cmc_avg['Market'][0])
            return sum(mAP)/len(mAP), cmc_avg
        else:
            return 0, {'Market': [0]*50}
        

def visualize_att_map(attention_maps, paths, save_dir='visualization_attention_maps'):
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    import PIL.Image as Image
    # paths[0] = query path, rest gallery paths
    query = cv2.imread(paths[0], 1)
    for path, attention_map in zip(paths[1:], attention_maps):
        gallery = cv2.imread(path, 1)
        attention_map = cv2.resize(attention_map.squeeze().cpu().numpy(), (gallery.shape[1], gallery.shape[0]))
        cam = show_cam_on_image(gallery, attention_map)        
        
        fig = figure(figsize=(6, 10), dpi=80)
        # Create figure and axes
        fig.add_subplot(1,3,1)
        plt.imshow(cv2.cvtColor(query, cv2.COLOR_BGR2RGB))

        fig.add_subplot(1,3,2)
        plt.imshow(cv2.cvtColor(gallery, cv2.COLOR_BGR2RGB))

        fig.add_subplot(1,3,3)
        plt.imshow(cv2.cvtColor(cam, cv2.COLOR_BGR2RGB))
        
        plt.axis('off')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, \
            os.path.basename(paths[0])[:-4]  + "_" +os.path.basename(path)))
    quit()

def show_cam_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)