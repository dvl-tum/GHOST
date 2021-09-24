from . import calc_mean_average_precision
import torch
import logging
import json
from collections import defaultdict
import sklearn.cluster
import sklearn.metrics.cluster
import os
from sklearn.manifold import TSNE

logger = logging.getLogger('GNNReID.Evaluator')

class Evaluator():
    def __init__(self, lamb, k1, k2, output_test_enc='norm', output_test_gnn='norm', re_rank=False, cat=0):
        self.lamb = lamb
        self.k1 = k1
        self.k2 = k2
        self.output_test = output_test_enc
        self.re_rank = re_rank

    def evaluate(self, model, dataloader, query=None, gallery=None, 
            gnn=None, graph_generator=None, dl_ev_gnn=None, net_type='bn_inception',
            dataroot='CARS', nb_classes=None, gallery_dl=None):
        model_is_training = model.training
        model.eval()

        if not gnn: # normal setting
            _, _, features, _ = self.predict_batchwise_reid(model, dataloader)
        else: #gnn
            gnn_is_training = gnn.training
            gnn.eval()
            _, _, features, _ = self.predict_batchwise_gnn(model, gnn,
                                                           graph_generator,
                                                           dataloader)
            gnn.train(gnn_is_training)
        mAP, cmc = calc_mean_average_precision(features, query, gallery,
                                               self.re_rank, self.lamb,
                                               self.k1, self.k2)
        model.train(model_is_training)
        return mAP, cmc

    def predict_batchwise_reid(self, model, dataloader):
        fc7s, L = [], []
        features = dict()
        labels = dict()

        with torch.no_grad():
            for X, Y, I, P in dataloader:
                if torch.cuda.is_available(): X = X.cuda()
                _, fc7 = model(X, output_option=self.output_test, val=True)
                if type(fc7) == list:
                    fc7 = torch.cat(fc7, dim=1)
                for path, out, y in zip(P, fc7, Y):
                    features[path] = out
                    labels[path] = y
                fc7s.append(fc7.cpu())
                L.append(Y)
        fc7, Y = torch.cat(fc7s), torch.cat(L)

        return torch.squeeze(fc7), torch.squeeze(Y), features, labels

    def predict_batchwise_gnn(self, model, gnn, graph_generator, dataloader):
        fc7s, L = [], []
        features = dict()
        labels = dict()
        with torch.no_grad():
            for X, Y, I, P in dataloader:
                if torch.cuda.is_available(): X = X.cuda()
                _, _, fc7 = model(X, output_option=self.output_test, val=True) ##### Actually _, fc7, _ CHECK THIS
                edge_attr, edge_index, fc7 = graph_generator.get_graph(fc7)
                _, fc7 = gnn(fc7, edge_index, edge_attr,
                             output_option=self.output_test)
                for path, out, y in zip(P, fc7, Y):
                    features[path] = out
                    labels[path] = y
                fc7s.append(fc7.cpu())
                L.append(Y)
        fc7, Y = torch.cat(fc7s), torch.cat(L)

        return torch.squeeze(fc7), torch.squeeze(Y), features, labels

