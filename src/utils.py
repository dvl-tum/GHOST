import motmetrics as mm
import logging
from sklearn.metrics import average_precision_score
import sklearn.metrics.pairwise
import numpy as np


logger = logging.getLogger('AllReIDTracker.Utils')

def evaluate_mot_accums(accums, names, generate_overall=False):
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accums,
        metrics=mm.metrics.motchallenge_metrics,
        names=names,
        generate_overall=generate_overall,)

    str_summary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,)
    logger.info('/n' + str_summary)

    return summary


def eval_metrics(X, y, topk=20):
        X, y = X.cpu(), y.cpu()
        dist = sklearn.metrics.pairwise.pairwise_distances(X)
        if type(dist) != np.ndarray:
            dist = dist.cpu().numpy()

        indices = np.argsort(dist, axis=1)
        matches = (y[indices] == y[:, np.newaxis])

        aps = []
        ret = np.zeros(topk)
        num_valid_queries = 0
        topk = 1
        for k in range(dist.shape[0]):
            # map
            y_true = matches[k, :]
            y_score = -dist[k][indices[k]]
            if not np.any(y_true): continue
            aps.append(average_precision_score(y_true, y_score))

            # rank
            index = np.nonzero(matches[i, :])[0]
            delta = 1. / len(index)
            for j, k in enumerate(index):
                if k - j >= topk: break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
            num_valid_queries += 1

        rank_1 = ret.cumsum() / num_valid_queries
        mAP = np.mean(aps)

        logger.info("Rank-1: {}, mAP: {}".format(rank_1, mAP))

        return rank_1, mAP
