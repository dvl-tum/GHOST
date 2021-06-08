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
    logger.info(str_summary)

    return summary


def eval_metrics(X, y, X_g=None, y_g=None, topk=20, first_match_break=True, gallery_mask=None):
        X, y = X.cpu().numpy(), y.cpu().numpy()
        if X_g is not None:
            X_g, y_g = X_g.cpu().numpy(), y_g.cpu().numpy()
        else:
            X_g = X
            y_g = y
        dist = sklearn.metrics.pairwise.pairwise_distances(X, X_g)
        if type(dist) != np.ndarray:
            dist = dist.cpu().numpy()
        #indices = np.argsort(dist, axis=1)
        #indices = indices[:, 1:]
        #matches = (y_g[indices] == y[:, np.newaxis])

        aps = []
        ret = np.zeros(topk)
        num_valid_queries = 0
        topk = 1
        for k in range(dist.shape[0]):
            # map
            if gallery_mask is not None:
                valid_dist = dist[k][gallery_mask[k]]
                valid_ys = y_g[gallery_mask[k]]
                indices = np.argsort(valid_dist)
            else:
                valid_dist = dist[k]
                valid_ys = y_g
                indices = np.argsort(valid_dist)[1:]

            y_true = (valid_ys[indices] == y[k])
            #y_true = matches[k, :]

            y_score = -valid_dist[indices] 
            if not np.any(y_true): continue
            aps.append(average_precision_score(y_true, y_score))

            # rank
            index = np.nonzero(y_true)[0]#matches[k, :])[0]
            delta = 1. / len(index)
            for j, i in enumerate(index):
                if i - j >= topk: break
                if first_match_break:
                    ret[i - j] += 1
                    break
                ret[i - j] += delta
            num_valid_queries += 1

        rank_1 = ret.cumsum() / num_valid_queries
        mAP = np.mean(aps)

        logger.info("Rank-1: {}, mAP: {}".format(rank_1, mAP))

        return rank_1, mAP


def update(oids, hids, dists, indices, events, m):
    """
        tracks : my results of shape {tr_id: {'id', 'im_index', 'max_iou', 'bbox'}
        num_frames : number of frames
    """
    import pandas as pd
    asso = dict()
    hypo = dict()
    cols = ['Type', 'id', 'frame', 'tr_id', 'iou', 'w', 'h']
    events = pd.DataFrame(columns=cols)
    for i in range(num_frames):
        for tr_id, tr in self.tracks.items():
            for t in tr:
                if t['im_index'] == i:
                    if t['id'] in asso.keys():
                        h_o = asso[t['id']]
                        if tr_id == h_o:
                            TYPE = 'match'
                        else:
                            if h_o in hypo.keys():
                                TYPE = 'switch'
                            else:
                                TYPE = 'ascend'
                    elif tr_id == -1:
                        TYPE = 'fp'
                    elif tr_id in hypo.keys():
                        if t['id'] not in asso.keys():
                            TYPE = 'migrate'
                        elif hypo[tr_id] != t['id']:
                            TYPE = 'transfer'
                    df = pd.DataFrame([[TYPE, t['id'], i, tr_id, t['iou'], t['bbox'][2]-t['bbox'][0], t['bbox'][3]-t['bbox'][1]]], columns=cols)

    #self.dirty_events = True
    oids = np.asarray(oids)
    oids_masked = np.zeros_like(oids, dtype=np.bool)
    hids = np.asarray(hids)
    hids_masked = np.zeros_like(hids, dtype=np.bool)
    dists = np.atleast_2d(dists).astype(float).reshape(oids.shape[0], hids.shape[0]).copy()

    if frameid is None:
        assert self.auto_id, 'auto-id is not enabled'
        if len(self._indices['FrameId']) > 0:
            frameid = self._indices['FrameId'][-1] + 1
        else:
            frameid = 0
    else:
        assert not self.auto_id, 'Cannot provide frame id when auto-id is enabled'

    eid = itertools.count()

    # 0. Record raw events

    no = len(oids)
    nh = len(hids)

    # Add a RAW event simply to ensure the frame is counted.
    indices.append(frameid, next(eid))
    events.append('RAW', np.nan, np.nan, np.nan)

    # There must be at least one RAW event per object and hypothesis.
    # Record all finite distances as RAW events.
    valid_i, valid_j = np.where(np.isfinite(dists))
    valid_dists = dists[valid_i, valid_j]
    for i, j, dist_ij in zip(valid_i, valid_j, valid_dists):
        indices.append(frameid, next(eid))
        events.append('RAW', oids[i], hids[j], dist_ij)
    # Add a RAW event for objects and hypotheses that were present but did
    # not overlap with anything.
    used_i = np.unique(valid_i)
    used_j = np.unique(valid_j)
    unused_i = np.setdiff1d(np.arange(no), used_i)
    unused_j = np.setdiff1d(np.arange(nh), used_j)
    for oid in oids[unused_i]:
        indices.append(frameid, next(eid))
        events.append('RAW', oid, np.nan, np.nan)
    for hid in hids[unused_j]:
        indices.append(frameid, next(eid))
        events.append('RAW', np.nan, hid, np.nan)

    if oids.size * hids.size > 0:
        # 1. Try to re-establish tracks from previous correspondences
        for i in range(oids.shape[0]):
            # No need to check oids_masked[i] here.
            if oids[i] not in self.m:
                continue

            hprev = self.m[oids[i]]
            j, = np.where(~hids_masked & (hids == hprev))
            if j.shape[0] == 0:
                continue
            j = j[0]

            if np.isfinite(dists[i, j]):
                o = oids[i]
                h = hids[j]
                oids_masked[i] = True
                hids_masked[j] = True
                self.m[oids[i]] = hids[j]

                indices.append(frameid, next(eid))
                events.append('MATCH', oids[i], hids[j], dists[i, j])
                self.last_match[o] = frameid
                self.hypHistory[h] = frameid

        # 2. Try to remaining objects/hypotheses
        dists[oids_masked, :] = np.nan
        dists[:, hids_masked] = np.nan

        rids, cids = linear_sum_assignment(dists)

        for i, j in zip(rids, cids):
            if not np.isfinite(dists[i, j]):
                continue

            o = oids[i]
            h = hids[j]
            is_switch = (o in self.m and
                            self.m[o] != h and
                            abs(frameid - self.last_occurrence[o]) <= self.max_switch_time)
            cat1 = 'SWITCH' if is_switch else 'MATCH'
            if cat1 == 'SWITCH':
                if h not in self.hypHistory:
                    subcat = 'ASCEND'
                    indices.append(frameid, next(eid))
                    events.append(subcat, oids[i], hids[j], dists[i, j])
            # ignore the last condition temporarily
            is_transfer = (h in self.res_m and
                            self.res_m[h] != o)
            # is_transfer = (h in self.res_m and
            #                self.res_m[h] != o and
            #                abs(frameid - self.last_occurrence[o]) <= self.max_switch_time)
            cat2 = 'TRANSFER' if is_transfer else 'MATCH'
            if cat2 == 'TRANSFER':
                if o not in self.last_match:
                    subcat = 'MIGRATE'
                    indices.append(frameid, next(eid))
                    events.append(subcat, oids[i], hids[j], dists[i, j])
                indices.append(frameid, next(eid))
                events.append(cat2, oids[i], hids[j], dists[i, j])
            if vf != '' and (cat1 != 'MATCH' or cat2 != 'MATCH'):
                if cat1 == 'SWITCH':
                    vf.write('%s %d %d %d %d %d\n' % (subcat[:2], o, self.last_match[o], self.m[o], frameid, h))
                if cat2 == 'TRANSFER':
                    vf.write('%s %d %d %d %d %d\n' % (subcat[:2], h, self.hypHistory[h], self.res_m[h], frameid, o))
            self.hypHistory[h] = frameid
            self.last_match[o] = frameid
            indices.append(frameid, next(eid))
            events.append(cat1, oids[i], hids[j], dists[i, j])
            oids_masked[i] = True
            hids_masked[j] = True
            self.m[o] = h
            self.res_m[h] = o

    # 3. All remaining objects are missed
    for o in oids[~oids_masked]:
        indices.append(frameid, next(eid))
        events.append('MISS', o, np.nan, np.nan)
        if vf != '':
            vf.write('FN %d %d\n' % (frameid, o))

    # 4. All remaining hypotheses are false alarms
    for h in hids[~hids_masked]:
        indices.append(frameid, next(eid))
        events.append('FP', np.nan, h, np.nan)
        if vf != '':
            vf.write('FP %d %d\n' % (frameid, h))

    # 5. Update occurance state
    for o in oids:
        self.last_occurrence[o] = frameid

    return frameid