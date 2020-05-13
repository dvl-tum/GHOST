import evaluation
import torch


def predict_batchwise_reid(model, dataloader):
    fc7s, L = [], []
    features = dict()
    labels = dict()
    with torch.no_grad():
        for X, Y, P in dataloader:
            _, fc7 = model(X.cuda())
            for path, out, y in zip(P, fc7, Y):
                features[path] = out
                labels[path] = y
            fc7s.append(fc7.cpu())
            L.append(Y)
    fc7, Y = torch.cat(fc7s), torch.cat(L)
    return torch.squeeze(fc7), torch.squeeze(Y), features, labels


def evaluate_reid(model, dataloader, query=None, gallery=None, root=None):
    model_is_training = model.training
    model.eval()
    _, _, features, _ = predict_batchwise_reid(model, dataloader)
    mAP, cmc = evaluation.calc_mean_average_precision(features, query,
                                                      gallery, root)
    model.train(model_is_training)
    return mAP, cmc

