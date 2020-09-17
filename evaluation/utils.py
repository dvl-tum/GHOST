from . import calc_mean_average_precision, calc_normalized_mutual_information, \
    cluster_by_kmeans, assign_by_euclidian_at_k, calc_recall_at_k
import torch
import logging
import json
from collections import defaultdict

logger = logging.getLogger('GNNReID.Evaluator')


class Evaluator_DML():
    def __init__(self, lamb, k1, k2, output_test='norm', re_rank=False):
        self.lamb = lamb
        self.k1 = k1
        self.k2 = k2
        self.output_test = output_test
        self.re_rank = re_rank

    def evaluate(self, model, dataloader, nb_classes, gnn=None,
                 graph_generator=None, dl_ev_gnn=None, net_type='bn_inception',
                 dataroot='CARS'):
        model_is_training = model.training
        model.eval()

        # calculate embeddings with model, also get labels (non-batch-wise)
        if not gnn:
            X, T = self.predict_batchwise(model, dataloader, net_type)
        elif dl_ev_gnn is not None:
            if dl_ev_gnn.dataset.labels_train is not None:  # traintest
                gnn_is_training = gnn.training
                gnn.eval()
                X, T = self.predict_batchwise_traintest(model, gnn,
                                                        graph_generator,
                                                        dataloader, dl_ev_gnn)
                gnn.train(gnn_is_training)
            else:  # pseudo
                gnn_is_training = gnn.training
                gnn.eval()
                X, T = self.predict_batchwise_pseudo(model, gnn,
                                                     graph_generator,
                                                     dataloader, dl_ev_gnn)
                gnn.train(gnn_is_training)
        else:  # gnn
            gnn_is_training = gnn.training
            gnn.eval()
            X, T = self.predict_batchwise_gnn(model, gnn, graph_generator,
                                              dataloader)
            gnn.train(gnn_is_training)

        if dataroot != 'Stanford':
            # calculate NMI with kmeans clustering
            nmi = calc_normalized_mutual_information(T, cluster_by_kmeans(X, nb_classes))
            logging.info("NMI: {:.3f}".format(nmi * 100))
        else:
            nmi = -1

        recall = []
        if dataroot != 'Stanford':
            Y = assign_by_euclidian_at_k(X, T, 8)
            which_nearest_neighbors = [1, 2, 4, 8]
        else:
            Y = assign_by_euclidian_at_k(X, T, 1000)
            which_nearest_neighbors = [1, 10, 100, 1000]

        for k in which_nearest_neighbors:
            r_at_k = calc_recall_at_k(T, Y, k)
            recall.append(r_at_k)
            logging.info("R@{} : {:.3f}".format(k, 100 * r_at_k))

        model.train(model_is_training)  # revert to previous training state
        return nmi, recall

    # just looking at this gives me AIDS, fix it fool!
    def predict_batchwise(self, model, dataloader, net_type):
        fc7s, L = [], []
        with torch.no_grad():
            for X, Y, _ in dataloader:
                if torch.cuda.is_available(): X = X.cuda()
                _, _, fc7 = model(X, output_option=self.output_test, val=True)
                fc7s.append(fc7.cpu())
                L.append(Y)
        fc7, Y = torch.cat(fc7s), torch.cat(L)
        return torch.squeeze(fc7), torch.squeeze(Y)

    def predict_batchwise_gnn(self, model, gnn, graph_generator, dataloader):
        fc7s, L = [], []
        with torch.no_grad():
            for X, Y, _ in dataloader:
                if torch.cuda.is_available(): X = X.cuda()
                _, _, fc7 = model(X, output_option=self.output_test,
                                  val=True)  ##### Actually _, fc7, _ CHECK THIS
                edge_attr, edge_index, fc7 = graph_generator.get_graph(fc7)
                _, fc7 = gnn(fc7, edge_index, edge_attr,
                             output_option=self.output_test)
                fc7s.append(fc7.cpu())
                L.append(Y)
        fc7, Y = torch.cat(fc7s), torch.cat(L)

        return torch.squeeze(fc7), torch.squeeze(Y)

    def predict_batchwise_pseudo(self, model, gnn, graph_generator, dataloader,
                                 dl_ev_gnn):
        fc7s, L = [], []
        preds = dict()
        features = dict()
        labels = dict()
        with torch.no_grad():
            for X, Y, P in dataloader:
                if torch.cuda.is_available(): X = X.cuda()
                pred, _, fc7 = model(X, output_option=self.output_test,
                                     val=True)
                for path, out, y, p in zip(P, fc7, Y, pred):
                    features[path] = out
                    preds[path] = torch.argmax(p).detach()
                    labels[path] = y

        for k, v in preds.items():
            ind = dl_ev_gnn.dataset.im_paths.index(k)
            dl_ev_gnn.dataset.ys[ind] = v.item()

        with torch.no_grad():
            for X, Y, P in dl_ev_gnn:
                fc7 = torch.stack([features[p] for p in P])
                edge_attr, edge_index, fc7 = graph_generator.get_graph(fc7)
                _, fc7 = gnn(fc7, edge_index, edge_attr,
                             output_option=self.output_test)

                fc7s.append(fc7.cpu())
                L.append(torch.tensor([labels[p] for p in P]))
        fc7, Y = torch.cat(fc7s), torch.cat(L)

        return torch.squeeze(fc7), torch.squeeze(Y)

    def predict_batchwise_traintest(self, model, gnn, graph_generator,
                                    dataloader, dl_ev_gnn):
        fc7s, L = [], []
        features = dict()
        with torch.no_grad():
            for X, Y, P in dataloader:
                if torch.cuda.is_available(): X = X.cuda()
                pred, _, fc7 = model(X, output_option=self.output_test,
                                     val=True)
                for path, out, y, p in zip(P, fc7, Y, pred):
                    features[path] = out

        logger.info("GNN")
        with torch.no_grad():
            # for i in len(dl_ev_gnn.l_inds)):
            for X, Y, P in dl_ev_gnn:
                if torch.cuda.is_available(): X = X.cuda()
                fc7 = torch.stack([features[p] for p in P])
                edge_attr, edge_index, fc7 = graph_generator.get_graph(fc7)
                _, fc7 = gnn(fc7, edge_index, edge_attr,
                             output_option=self.output_test)
                fc7s.append(fc7.cpu())
                L.append(Y)
            fc7, Y = torch.cat(fc7s), torch.cat(L)

        return torch.squeeze(fc7), torch.squeeze(Y)


class Evaluator():
    def __init__(self, lamb, k1, k2, output_test='norm', re_rank=False):
        self.lamb = lamb
        self.k1 = k1
        self.k2 = k2
        self.output_test = output_test
        self.re_rank = re_rank

    def evaluate_reid(self, model, dataloader, query=None, gallery=None, 
            gnn=None, graph_generator=None, dl_ev_gnn=None):
        model_is_training = model.training
        model.eval()

        if not gnn: # normal setting
            _, _, features, _ = self.predict_batchwise_reid(model, dataloader)
        elif dl_ev_gnn is not None: # either pseudo or traintest
            if dl_ev_gnn.dataset.labels_train is not None: # traintest
                gnn_is_training = gnn.training
                gnn.eval()
                _, _, features, _ = self.predict_batchwise_traintest(model, gnn,
                                                                  graph_generator,
                                                                  dataloader,
                                                                  dl_ev_gnn)
                gnn.train(gnn_is_training)
            else: # pseudo
                gnn_is_training = gnn.training
                gnn.eval()
                _, _, features, _ = self.predict_batchwise_pseudo(model, gnn,
                                                               graph_generator,
                                                               dataloader, dl_ev_gnn)
                gnn.train(gnn_is_training)
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
            for X, Y, P in dataloader:
                if torch.cuda.is_available(): X = X.cuda()
                _, _, fc7 = model(X, output_option=self.output_test, val=True)
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

    def predict_batchwise_pseudo(self, model, gnn, graph_generator, dataloader, dl_ev_gnn):
        fc7s, L = [], []
        preds = dict()
        features = dict()
        labels = dict()
        with torch.no_grad():
            for X, Y, P in dataloader:
                if torch.cuda.is_available(): X = X.cuda()
                pred, _, fc7 = model(X, output_option=self.output_test, val=True)
                for path, out, y, p in zip(P, fc7, Y, pred):
                    features[path] = out
                    preds[path] = torch.argmax(p).detach()
                    labels[path] = y

        for k, v in preds.items():
            ind = dl_ev_gnn.dataset.im_paths.index(k)
            dl_ev_gnn.dataset.ys[ind] = v.item()
            
        features_new = dict()
        labels_new = dict()
        with torch.no_grad():
            for X, Y, P in dl_ev_gnn:
                fc7 = torch.stack([features[p] for p in P])
                edge_attr, edge_index, fc7 = graph_generator.get_graph(fc7)
                _, fc7 = gnn(fc7, edge_index, edge_attr,
                             output_option=self.output_test)
                for path, out, y in zip(P, fc7, Y):
                    features_new[path] = out
                    labels_new[path] = y
                fc7s.append(fc7.cpu())
                L.append(Y)
        fc7, Y = torch.cat(fc7s), torch.cat(L)

        return torch.squeeze(fc7), torch.squeeze(Y), features_new, labels

    def predict_batchwise_traintest(self, model, gnn, graph_generator,
                                    dataloader, dl_ev_gnn):
        features = dict()
        with torch.no_grad():
            for X, Y, P in dataloader:
                if torch.cuda.is_available(): X = X.cuda()
                pred, _, fc7 = model(X, output_option=self.output_test, val=True)
                for path, out, y, p in zip(P, fc7, Y, pred):
                    features[path] = out

        logger.info("GNN")
        features_new = defaultdict(dict)
        with torch.no_grad():
            #for i in len(dl_ev_gnn.l_inds)):
            i = 0
            for X, Y, P in dl_ev_gnn:
                if torch.cuda.is_available(): X = X.cuda()
                fc7 = torch.stack([features[p] for p in P])
                edge_attr, edge_index, fc7 = graph_generator.get_graph(fc7)
                _, fc7 = gnn(fc7, edge_index, edge_attr,
                             output_option=self.output_test)
                #features_new[P[-1]][P[-2]] = {P[-1]: fc7[-1], P[-2]: fc7[-2]}
                features_new[P[-1]][P[-2]] = (fc7[-1] @ fc7[-1] + fc7[-2] @ fc7[-2] - 2 * (fc7[-1] @ fc7[-2])).detach()
                if i % 10000 == 0:
                    logger.info(i)
                i += 1
        return _, _, features_new, _
