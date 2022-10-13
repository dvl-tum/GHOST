from . import calc_mean_average_precision
import torch
import logging
import numpy as np

logger = logging.getLogger('GNNReID.Evaluator')

class Evaluator():
    def __init__(self, output_test_enc='norm'):
        self.output_test = output_test_enc

    def evaluate(self, model, dataloader, query=None, gallery=None, 
                    add_dist=False):

        model_is_training = model.training
        model.eval()

        logger.info('evaluate using only backbone')
        features, labels, camids = self.predict_reid_features_bb(
            model, dataloader, add_dist)

        if len(camids):
            logger.info("Using MOT17 - use camids and labels from dataset")
            qc = np.array([camids[k] for k in query])
            gc = np.array([camids[k] for k in gallery])
            qi = np.array([labels[k] for k in query])
            gi = np.array([labels[k] for k in gallery])
        else:
            gc = qc = gi = qi = None

        mAP, cmc = calc_mean_average_precision(features, query, gallery,
                                            gc=gc, qc=qc, gi=gi, qi=qi) 
        model.train(model_is_training)
        return mAP, cmc

    def predict_reid_features_bb(self, model, dataloader, add_dist):
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

                if torch.cuda.is_available():
                    X = X.cuda()

                if add_dist:
                    _, fc7, _ = model(
                        X, output_option=self.output_test, val=True)
                else:
                    _, fc7 = model(
                        X, output_option=self.output_test, val=True)

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
