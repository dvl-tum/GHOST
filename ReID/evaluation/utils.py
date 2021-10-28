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
            query_guided=False, dl_ev_gnn=None, queryguided=False):
        # query_guided == use QG network
        # queryguided == evaluate for each gallery and each query
        model_is_training = model.training
        model.eval()

        # batchwise evaluation
        if (batchwise_rank or query_guided) and not queryguided and gnn:
            gnn_is_training = gnn.training
            gnn.eval()
            mAP, cmc = self.predict_batchwise_gnn(model, gnn,
                                                        graph_generator,
                                                        dataloader,
                                                        batchwise_rank=batchwise_rank,
                                                        query_guided=query_guided)
            gnn.train(gnn_is_training)
        
        # query guided evaluation
        elif query_guided and queryguided and gnn:
            gnn_is_training = gnn.training
            gnn.eval()
            mAP, cmc = self.predict_batchwise_gnn_queryguided(model, gnn,
                                                        graph_generator,
                                                        dataloader,
                                                        batchwise_rank=batchwise_rank,
                                                        query_guided=query_guided,
                                                        dataloader_queryguided=dl_ev_gnn,
                                                        query=query, gallery=gallery)
            gnn.train(gnn_is_training)
        else:
            # normal setting
            if not gnn: 
                _, _, features, _ = self.predict_batchwise_reid(model, dataloader, add_dist)
            #gnn with sampled batches
            else: 
                gnn_is_training = gnn.training
                gnn.eval()
                features, _ = self.predict_batchwise_gnn(model, gnn,
                                                            graph_generator,
                                                            dataloader)
                gnn.train(gnn_is_training)

            mAP, cmc = calc_mean_average_precision(features, query, gallery,
                                                self.re_rank, self.lamb,
                                               self.k1, self.k2) 
        model.train(model_is_training)
        return mAP, cmc

    def predict_batchwise_reid(self, model, dataloader, add_dist, attention=True):
        fc7s, L = [], []
        features = dict()
        labels = dict()

        with torch.no_grad():
            for X, Y, I, P in dataloader:
                if torch.cuda.is_available(): X = X.cuda()
                if add_dist:
                    # p_score = person score
                    _, fc7, p_score = model(X, output_option=self.output_test, val=True)
                elif attention:
                    _, fc7, _ = model(X, output_option=self.output_test, val=True)
                else:
                    _, fc7 = model(X, output_option=self.output_test, val=True)
                if type(fc7) == list:
                    fc7 = torch.cat(fc7, dim=1)
                for path, out, y in zip(P, fc7, Y):
                    features[path] = out.detach()
                    labels[path] = y.detach()
                fc7s.append(fc7.detach().cpu())
                L.append(Y.detach())
        fc7, Y = torch.cat(fc7s), torch.cat(L)

        return torch.squeeze(fc7), torch.squeeze(Y), features, labels

    def predict_batchwise_gnn(self, model, gnn, graph_generator, dataloader, \
        attention=True, batchwise_rank=False, query_guided=False):
        features = dict()
        labels = dict()
        CMC, mAP = list(), list()
        with torch.no_grad():
            for X, Y, I, P in dataloader:
                if torch.cuda.is_available(): X = X.cuda()

                # get backbone features
                if attention:
                    _, queries, fc7 = model(X, output_option=self.output_test, val=True)
                else:
                    _, fc7 = model(X, output_option=self.output_test, val=True) 
                
                if not query_guided:
                    edge_attr, edge_index, fc7 = graph_generator.get_graph(fc7)
                    _, fc7, _ = gnn(fc7, edge_index, edge_attr,
                                    output_option=self.output_test)
                
                else:
                    _, _, _, qs, gs, attended_feats, fc_x = gnn(fc7)
                    #logger.info(attended_feats.shape)
                    attended_feats = attended_feats.reshape(X.shape[0], X.shape[0]-1, -1)
                    #logger.info(attended_feats.shape)
                    #logger.info(gs, qs)
                    dist = [sklearn.metrics.pairwise_distances(q.unsqueeze(0).cpu().numpy(), \
                        f.cpu().numpy(), metric='euclidean') for q, f in zip(queries, attended_feats)]
                    dist = np.asarray([np.insert(d, i, 0) for i, d in enumerate(dist)])

                for path, out, y in zip(P, fc7[-1], Y):
                    features[path] = out.detach()
                    labels[path] = y.detach()
                
                if batchwise_rank or query_guided:
                    if not query_guided:
                        qg = list(features.keys())
                        maP, cmc = calc_mean_average_precision(features.detach(), qg, qg)
                    else:
                        maP, cmc = calc_mean_average_precision(None, P, P, distmat=dist)
                    #logger.info(maP)
                    #logger.info(cmc['Market'][0])
                    #quit()
                    CMC.append(cmc)
                    mAP.append(maP)

                    features, labels = dict(), dict()

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
        
        return features, labels

    def predict_batchwise_gnn_queryguided(self, model, gnn, graph_generator, dataloader, \
            attention=True, batchwise_rank=False, query_guided=False, query=None, \
            gallery=None, dataloader_queryguided=None):
        features = dict()
        feature_maps = dict()
        CMC, mAP = list(), list()
        indices = dict()
        with torch.no_grad():
            for X, Y, I, P in dataloader:
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
                qg_dists = dict()
                for Y, I, P in dataloader_queryguided:
                    # get feature maps
                    fc7 = torch.cat([feature_maps[p].unsqueeze(0) for p in P], 0)

                    # get attended features
                    _, _, _, qs, gs, attended_feats, _ = gnn(fc7, num_query=1)

                    # get distance
                    dist = sklearn.metrics.pairwise_distances(features[P[0]].unsqueeze(0).cpu().numpy(), \
                        attended_feats.cpu().numpy(), metric='euclidean')
                    
                    for d, p in zip(dist[0], P[1:]):
                        qg_dists[p] = d
                    
                dist = np.asarray([v for v in qg_dists.values()])
                gallery = [k for k in qg_dists.keys()]
                query = [P[0]]
                maP, cmc = calc_mean_average_precision(None, query=query, gallery=gallery, distmat=dist)
                #quit()
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
        

