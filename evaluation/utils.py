from . import calc_mean_average_precision
import torch
import logging
import json

logger = logging.getLogger('GNNReID.Evaluator')


class Evaluator():
    def __init__(self, lamb, k1, k2, output_test='norm',
                 re_rank=False):
        self.lamb = lamb
        self.k1 = k1
        self.k2 = k2
        self.output_test = output_test
        self.re_rank = re_rank

    def evaluate_reid(self, model, dataloader, query=None, gallery=None, 
            gnn=None, graph_generator=None, dl_ev_gnn=None):
        model_is_training = model.training
        model.eval()
        if not gnn:
            _, _, features, _ = self.predict_batchwise_reid(model, dataloader)
        elif dl_ev_gnn:
            gnn_is_training = gnn.training
            gnn.eval()
            _, _, features, _ = self.predict_batchwise_pseudo(model, gnn,
                                                           graph_generator,
                                                           dataloader, dl_ev_gnn)
            gnn.train(gnn_is_training)
        else:
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
            for X, Y, P in dataloader:
                if torch.cuda.is_available(): X = X.cuda()
                _, _, fc7 = model(X, output_option=self.output_test)
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
            for X, Y, P in dataloader:
                if torch.cuda.is_available(): X = X.cuda()
                _, _, fc7 = model(X, output_option=self.output_test) ##### Actually _, fc7, _ CHECK THIS
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

    def predict_batchwise_pseudo(self, model, gnn, graph_generator, dataloader, dl_ev_gnn):
        fc7s, L = [], []
        preds = dict()
        features = dict()
        labels = dict()
        with torch.no_grad():
            for X, Y, P in dataloader:
                if torch.cuda.is_available(): X = X.cuda()
                pred, _, fc7 = model(X, output_option=self.output_test)
                for path, out, y in zip(P, fc7, Y):
                    features[path] = fc7
                    preds[path] = torch.argmax(pred).detach()
                    labels[path] = y

            for k, v in preds.items():
                ind = dl_ev_gnn.dataset.im_paths.index([k])
                dl_ev_gnn.dataset.ys[ind] = v

            for X, Y, P in dl_ev_gnn:
                fc7 = torch.stack([features[p] for p in P])
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
