import evaluation
import torch
import net
import data_utility


def predict_batchwise_reid(model, dataloader):
    fc7s, L = [], []
    features = dict()
    labels = dict()
    with torch.no_grad():
        for X, Y, P in dataloader:
            if torch.cuda.is_available(): X = X.cuda()
            _, fc7 = model(X)
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

if __name__ == '__main__':

    model = net.load_net(dataset='cuhk03',
                         net_type='resnet50',
                         nb_classes=751,
                         embed=False,
                         sz_embedding=512,
                         pretraining=True)

    dl_tr, dl_ev, q, g = data_utility.create_loaders(
        data_root='../../datasets/cuhk03-np/detected',
        input_size=224, size_batch=4, pretraining=False,
        num_workers=2, num_classes_iter=2, num_elements_class=2
        )

    evaluate_reid(model, dl_ev, q, g, '../../datasets/cuhk03-np/detected')
