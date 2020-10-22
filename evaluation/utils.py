from . import calc_mean_average_precision, calc_normalized_mutual_information, \
    cluster_by_kmeans, assign_by_euclidian_at_k, calc_recall_at_k
import torch
import logging
import json
from collections import defaultdict
import sklearn.cluster
import sklearn.metrics.cluster

logger = logging.getLogger('GNNReID.Evaluator')


class Evaluator_DML():
    def __init__(self, lamb, k1, k2, output_test_enc='norm', output_test_gnn='norm', re_rank=False, cat=0, nb_clusters=0):
        self.nb_clusters = nb_clusters
        self.lamb = lamb
        self.k1 = k1
        self.k2 = k2
        self.output_test_enc = output_test_enc
        self.output_test_gnn = output_test_gnn
        self.re_rank = re_rank
        self.cat = cat

    def evaluate(self, model, dataloader, query=None, gallery=None,
            gnn=None, graph_generator=None, dl_ev_gnn=None, net_type='bn_inception',
            dataroot='CARS', nb_classes=None):
        self.dataroot = dataroot
        self.nb_classes = nb_classes
        model_is_training = model.training
        model.eval()

        # calculate embeddings with model, also get labels (non-batch-wise)
        if not gnn:
            X, T = self.predict_batchwise(model, dataloader, net_type)
        elif dl_ev_gnn is not None:
            if dl_ev_gnn.sampler.__class__.__name__ == 'PseudoSampler' or dl_ev_gnn.sampler.__class__.__name__ == 'PseudoSamplerV' or dl_ev_gnn.sampler.__class__.__name__ == 'PseudoSamplerIV':
                gnn_is_training = gnn.training
                gnn.eval()
                logger.info("Evaluate KNN evaluate")
                X, T = self.predict_batchwise_knn(model, gnn,
                                                        graph_generator,
                                                        dataloader, dl_ev_gnn)
                gnn.train(gnn_is_training)

            elif dl_ev_gnn.dataset.labels_train is not None:  # traintest
                gnn_is_training = gnn.training
                gnn.eval()
                X, T = self.predict_batchwise_traintest(model, gnn,
                                                        graph_generator,
                                                        dataloader, dl_ev_gnn)
                gnn.train(gnn_is_training)
            else:  # pseudo
                logger.info("Using {} number of clusters for kmeans sampling".format(self.nb_clusters))
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

        if dataroot != 'Stanford_Online_Products':
            # calculate NMI with kmeans clustering
            nmi = calc_normalized_mutual_information(T, cluster_by_kmeans(X, nb_classes))
            logger.info("NMI: {:.3f}".format(nmi * 100))
        else:
            nmi = -1

        recall = []
        if dataroot != 'Stanford_Online_Products':
            Y, T = assign_by_euclidian_at_k(X, T, 8)
            which_nearest_neighbors = [1, 2, 4, 8]
        else:
            Y, T = assign_by_euclidian_at_k(X, T, 1000)
            which_nearest_neighbors = [1, 10, 100, 1000]

        for k in which_nearest_neighbors:
            r_at_k = calc_recall_at_k(T, Y, k)
            recall.append(r_at_k)
            logger.info("R@{} : {:.3f}".format(k, 100 * r_at_k))

        model.train(model_is_training)  # revert to previous training state
        return nmi, recall

    # just looking at this gives me AIDS, fix it fool!
    def predict_batchwise(self, model, dataloader, net_type):
        logger.info("Evaluate normal")
        fc7s, L = [], []
        with torch.no_grad():
            for X, Y, _, _ in dataloader:
                if torch.cuda.is_available(): X = X.cuda()
                _, fc7, _ = model(X, output_option=self.output_test_enc, val=True)
                fc7s.append(fc7.cpu())
                L.append(Y)
        fc7, Y = torch.cat(fc7s), torch.cat(L)
        return torch.squeeze(fc7), torch.squeeze(Y)

    def predict_batchwise_gnn(self, model, gnn, graph_generator, dataloader):
        logger.info("Evaluate gnn")
        fc7s, L = [], []
        feature_dict = dict()
        ys = dict()
        with torch.no_grad():
            for X, Y, I, P in dataloader:
                if torch.cuda.is_available(): X = X.cuda()
                _, fc7, _ = model(X, output_option=self.output_test_enc,
                                  val=True)  ##### Actually _, fc7, _ CHECK THIS
                for path, out, y, i in zip(P, fc7, Y, I):
                    feature_dict[i] = out
                    ys[i] = y

                edge_attr, edge_index, fc7 = graph_generator.get_graph(fc7)
                fc7 = fc7.cuda(0)
                edge_attr = edge_attr.cuda(0)
                edge_index = edge_index.cuda(0) 
                _, fc7 = gnn(fc7, edge_index, edge_attr,
                             output_option=self.output_test_gnn)
                
                if self.cat:
                    fc7 = torch.cat(fc7, dim=1)
                else:
                    fc7 = fc7[-1]

                fc7s.append(fc7.cpu())
                L.append(Y)
        fc7, Y = torch.cat(fc7s), torch.cat(L)
        
        # Evaliation after ResNet
        if self.dataroot != 'Stanford_Online_Products':
            x = torch.cat([f.unsqueeze(0).cpu() for f in feature_dict.values()], 0)
            ys = torch.cat([y.unsqueeze(0).cpu() for y in ys.values()], 0)
            cluster = sklearn.cluster.KMeans(self.nb_classes).fit(x).labels_
            NMI = sklearn.metrics.cluster.normalized_mutual_info_score(cluster, ys)
            logger.info("NMI after ResNet50 {}".format(NMI))
        
            y, ys = assign_by_euclidian_at_k(x, ys, 1)
            r_at_k = calc_recall_at_k(ys, y, 1)
            logger.info("R@{} after ResNet50: {:.3f}".format(1, 100 * r_at_k))

        return torch.squeeze(fc7), torch.squeeze(Y)

    def predict_batchwise_pseudo_rand(self, model, gnn, graph_generator, dataloader, dl_ev_gnn):
        fc7s, L = [], []
        logger.info("Evaluate Rand Pseudo")
        with torch.no_grad():
            for X, Y, _, _ in dataloader:
                if torch.cuda.is_available(): X = X.cuda()
                _, fc7, _ = model(X, output_option=self.output_test_enc,
                                  val=True)  ##### Actually _, fc7, _ CHECK THIS
                edge_attr, edge_index, fc7 = graph_generator.get_graph(fc7)
                _, fc7 = gnn(fc7, edge_index, edge_attr,
                             output_option=self.output_test_gnn)
                fc7s.append(fc7.cpu())
                L.append(Y)
        fc7, Y = torch.cat(fc7s), torch.cat(L)

        return torch.squeeze(fc7), torch.squeeze(Y)
    
    def predict_batchwise_pseudo(self, model, gnn, graph_generator, dataloader,
                                 dl_ev_gnn):
        logger.info("Evaluate Pseudo")
        fc7s, L = [], []
        preds = dict()
        features = dict()
        labels = dict()
        feature_dict = dict()
        features_dict = dict()
        ys = dict()
        with torch.no_grad():
            for X, Y, I, P in dataloader:
                if torch.cuda.is_available(): X = X.cuda()
                pred, fc7, _ = model(X, output_option=self.output_test_enc,
                                     val=True)
                for path, out, y, p, i in zip(P, fc7, Y, pred, I):
                    feature_dict[i] = out
                    features[path] = out
                    preds[path] = torch.argmax(p).detach()
                    labels[path] = y
                    ys[i] = y
                for y, f, i in zip(Y, fc7, I):
                    new_ind = y.data.item()-100
                    if new_ind in features_dict.keys():
                        features_dict[new_ind][i.item()] = f
                    else:
                        features_dict[new_ind] = {i.item(): f}
        
        logger.info("ResNet completed")
        
        # Evaliation after ResNet
        if self.dataroot != 'Stanford_Online_Products':
            x = torch.cat([f.unsqueeze(0).cpu() for f in feature_dict.values()], 0)
            ys = torch.cat([y.unsqueeze(0).cpu() for y in ys.values()], 0)
            logger.info("Compute KMeans for nb classes {}".format(self.nb_classes))
            cluster = sklearn.cluster.KMeans(self.nb_classes).fit(x).labels_
            logger.info("Compute NMI")
            NMI = sklearn.metrics.cluster.normalized_mutual_info_score(cluster, ys)
            logger.info("NMI after ResNet50 {}".format(NMI))
            logger.info("Compute Rand Index")
            RI = sklearn.metrics.adjusted_rand_score(ys, cluster)
            logger.info("RI after Resnet50 {}".format(RI))
            Y, ys = assign_by_euclidian_at_k(x, ys, 1)
            r_at_k = calc_recall_at_k(ys, Y, 1)
            logger.info("R@{} after ResNet50: {:.3f}".format(1, 100 * r_at_k))
        
        # Update after feature dict for sampling
        dl_ev_gnn.sampler.feature_dict = feature_dict
        dl_ev_gnn.sampler.nb_clusters = self.nb_clusters
        logger.info(dl_ev_gnn.sampler.nb_clusters)
        #dl_ev_gnn.sampler.feature_dict = features_dict

        features_new = dict()
        labels_new = dict()
        with torch.no_grad():
            for X, Y, I, P in dl_ev_gnn:
                fc7 = torch.stack([features[p] for p in P])
                edge_attr, edge_index, fc7 = graph_generator.get_graph(fc7)
                fc7 = fc7.cuda(0)
                edge_attr = edge_attr.cuda(0)
                edge_index = edge_index.cuda(0)
                _, fc7 = gnn(fc7, edge_index, edge_attr,
                             output_option=self.output_test_gnn)
                if self.cat:
                    fc7 = torch.cat(fc7, dim=1)
                else:
                    fc7 = fc7[-1]
                for path, out, y, i in zip(P, fc7, Y, I):
                    features_new[path] = out
                    labels_new[path] = labels[path]

        fc7 = torch.cat([v.unsqueeze(dim=0).cpu() for v in features_new.values()])
        Y = torch.cat([v.unsqueeze(dim=0).cpu() for v in labels_new.values()])
        return torch.squeeze(fc7), torch.squeeze(Y)
    
    def predict_batchwise_knn(self, model, gnn, graph_generator,
            dataloader, dl_ev_gnn):
        logger.info("KNN")
        fc7s, L = [], []
        preds = dict()
        features = dict()
        labels = dict()
        feature_dict = dict()
        features_dict = dict()
        ys = dict()
        with torch.no_grad():
            for X, Y, I, P in dataloader:
                if torch.cuda.is_available(): X = X.cuda()
                pred, fc7, _ = model(X, output_option=self.output_test_enc,
                                     val=True)
                for path, out, y, p, i in zip(P, fc7, Y, pred, I):
                    features[path] = out
                    feature_dict[i.item()] = out
                    preds[path] = torch.argmax(p).detach()
                    labels[path] = y #.data.item()
                    ys[i] = y
                for y, f, i in zip(Y, fc7, I):
                    if y.data.item() in features_dict.keys():
                        features_dict[y.data.item()][i.item()] = f.cpu()
                    else:
                        features_dict[y.data.item()] = {i.item(): f.cpu()}
        #print(features_dict)
        # Evaliation after ResNet
        if self.dataroot != 'Stanford_Online_Products':
            x = torch.cat([f.unsqueeze(0).cpu() for f in feature_dict.values()], 0)
            ys = torch.cat([y.unsqueeze(0).cpu() for y in ys.values()], 0)
            cluster = sklearn.cluster.KMeans(self.nb_classes).fit(x).labels_
            NMI = sklearn.metrics.cluster.normalized_mutual_info_score(cluster, ys)
            logger.info("KNN: NMI after ResNet50 {}".format(NMI))
            
            RI = sklearn.metrics.adjusted_rand_score(ys, cluster)
            logger.info("RI after Resnet50 {}".format(RI))

            Y, ys = assign_by_euclidian_at_k(x, ys, 1)
            r_at_k = calc_recall_at_k(ys, Y, 1)
            logger.info("KNN: R@{} after ResNet50: {:.3f}".format(1, 100 * r_at_k))
        
        # Update after feature dict for sampling
        dl_ev_gnn.sampler.feature_dict = feature_dict
        if dl_ev_gnn.sampler.__class__.__name__ == 'PseudoSampler':
            features_new = defaultdict(dict)
            labels_new = defaultdict(dict)
        elif dl_ev_gnn.sampler.__class__.__name__ == 'PseudoSamplerV' or dl_ev_gnn.sampler.__class__.__name__ == 'PseudoSamplerIV':
            features_new = dict()
            labels_new = dict()
            if dl_ev_gnn.sampler.__class__.__name__ == 'PseudoSamplerIV':
                dl_ev_gnn.sampler.feature_dict = features_dict
                
        gt = list() 
        with torch.no_grad():
            for X, Y, I, P in dl_ev_gnn:
                if torch.cuda.is_available(): X = X.cuda()
                fc7 = torch.stack([features[p] for p in P])
                edge_attr, edge_index, fc7 = graph_generator.get_graph(fc7)
                fc7 = fc7.cuda(0)
                edge_attr = edge_attr.cuda(0)
                edge_index = edge_index.cuda(0)
                _, fc7 = gnn(fc7, edge_index, edge_attr,
                             output_option=self.output_test_gnn)
                #features_new[P[-1]][P[-2]] = {P[-1]: fc7[-1], P[-2]: fc7[-2]}
                
                if self.cat:
                    fc7 = torch.cat(fc7, dim=1)
                else:
                    fc7 = fc7[-1]
                
                if dl_ev_gnn.sampler.__class__.__name__ == 'PseudoSampler':
                    p_0 = P[0]
                    out_0 = fc7[0]
                    label_0 = Y[0]
                    for p, out, y in zip(P, fc7, Y):
                        features_new[p_0][p] = (out_0 @ out_0 + out @ out - 2 * (out_0 @ out)).detach()
                        labels_new[p_0][p] = y
                
                elif dl_ev_gnn.sampler.__class__.__name__ == 'PseudoSamplerV' or dl_ev_gnn.sampler.__class__.__name__ == 'PseudoSamplerIV' :
                    anchors = [i for i in range(dl_ev_gnn.sampler.bs) if i%dl_ev_gnn.sampler.num_classes == 0]
                    for i in anchors:
                        labels_new[P[i]]= labels[P[i]]
                        features_new[P[i]]= fc7[i]

        if dl_ev_gnn.sampler.__class__.__name__ == 'PseudoSamplerV' or dl_ev_gnn.sampler.__class__.__name__ == 'PseudoSamplerIV':
            features_new = torch.cat([v.unsqueeze(dim=0).cpu() for v in features_new.values()]).squeeze()
            labels_new = torch.cat([v.unsqueeze(dim=0).cpu() for v in labels_new.values()]).squeeze()
        
        return features_new, labels_new
    
    def predict_batchwise_traintest(self, model, gnn, graph_generator,
                                    dataloader, dl_ev_gnn):
        features = dict()
        with torch.no_grad():
            for X, Y, P in dataloader:
                if torch.cuda.is_available(): X = X.cuda()
                pred, fc7, _ = model(X, output_option=self.output_test_enc, val=True)
                for path, out, y, p in zip(P, fc7, Y, pred):
                    features[path] = out

        logger.info("GNN")
        features_new = defaultdict(dict)
        with torch.no_grad():
            #for i in len(dl_ev_gnn.l_inds)):
            i = 0
            for X, Y, I, P in dl_ev_gnn:
                if torch.cuda.is_available(): X = X.cuda()
                fc7 = torch.stack([features[p] for p in P])
                edge_attr, edge_index, fc7 = graph_generator.get_graph(fc7)
                _, fc7 = gnn(fc7, edge_index, edge_attr,
                             output_option=self.output_test_gnn)
                #features_new[P[-1]][P[-2]] = {P[-1]: fc7[-1], P[-2]: fc7[-2]}
                features_new[P[-1]][P[-2]] = (fc7[-1] @ fc7[-1] + fc7[-2] @ fc7[-2] - 2 * (fc7[-1] @ fc7[-2])).detach()
                if i % 10000 == 0:
                    logger.info(i)
                i += 1
        return _, _, features_new, _


class Evaluator():
    def __init__(self, lamb, k1, k2, output_test='norm', re_rank=False):
        self.lamb = lamb
        self.k1 = k1
        self.k2 = k2
        self.output_test = output_test
        self.re_rank = re_rank

    def evaluate(self, model, dataloader, query=None, gallery=None, 
            gnn=None, graph_generator=None, dl_ev_gnn=None, net_type='bn_inception',
            dataroot='CARS', nb_classes=None):
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
            for X, Y, I, P in dataloader:
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

    def predict_batchwise_pseudo_rand(self, model, gnn, graph_generator, dataloader, dl_ev_gnn):
        fc7s, L = [], []
        features = dict()
        with torch.no_grad():
            for X, Y, I, P in dataloader:
                if torch.cuda.is_available(): X = X.cuda()
                _, _, fc7 = model(X, output_option=self.output_test,
                                  val=True)  ##### Actually _, fc7, _ CHECK THIS
                edge_attr, edge_index, fc7 = graph_generator.get_graph(fc7)
                _, fc7 = gnn(fc7, edge_index, edge_attr,
                             output_option=self.output_test)
                for path, out in zip(P, fc7):
                    features[path] = out

                fc7s.append(fc7.cpu())
                L.append(Y)
        fc7, Y = torch.cat(fc7s), torch.cat(L)

        return _, _, features, _
        
    def predict_batchwise_pseudo(self, model, gnn, graph_generator, dataloader, dl_ev_gnn):
        fc7s, L = [], []
        preds = dict()
        features = dict()
        labels = dict()
        with torch.no_grad():
            for X, Y, I, P in dataloader:
                if torch.cuda.is_available(): X = X.cuda()
                pred, _, fc7 = model(X, output_option=self.output_test, val=True)
                for path, out, y, p in zip(P, fc7, Y, pred):
                    features[path] = out
                    preds[path] = torch.argmax(p).detach()
                    labels[path] = y

        for k, v in preds.items():
            ind = dl_ev_gnn.dataset.im_paths.index(k)
            dl_ev_gnn.dataset.ys[ind] = v.item()
        
        ddict = defaultdict(list)
        for idx, label in enumerate(dl_ev_gnn.dataset.ys):
            ddict[label].append(idx)

        list_of_indices_for_each_class = []
        for key in ddict:
            list_of_indices_for_each_class.append(ddict[key])

        dl_ev_gnn.sampler.l_inds = list_of_indices_for_each_class

        features_new = dict()
        labels_new = dict()
        with torch.no_grad():
            for X, Y, I, P in dl_ev_gnn:
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
            for X, Y, I, P in dl_ev_gnn:
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
