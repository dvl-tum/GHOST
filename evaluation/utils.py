from . import calc_mean_average_precision
import torch

class Evaluator():
    def __init__(self, lamb, k1, k2, output_test='norm',
                 re_rank=False):
        self.lamb = lamb
        self.k1 = k1
        self.k2 = k2
        self.output_test = output_test
        self.re_rank = re_rank

    def evaluate_reid(self, model, dataloader, query=None, gallery=None):
        model_is_training = model.training
        model.eval()

        _, _, features, _ = self.predict_batchwise_reid(model, dataloader)
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
                _, fc7 = model(X, output_option=self.output_test)
                for path, out, y in zip(P, fc7, Y):
                    features[path] = out
                    labels[path] = y
                fc7s.append(fc7.cpu())
                L.append(Y)
        fc7, Y = torch.cat(fc7s), torch.cat(L)

        return torch.squeeze(fc7), torch.squeeze(Y), features, labels

