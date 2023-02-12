from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict
from scipy import stats
import os.path as osp
import os
import copy
from scipy.spatial import distance
import scipy.stats
import math
import logging
import argparse


logger = logging.getLogger('AllReIDTracker')
logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(message)s')
formatter = logging.Formatter('%(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(ch)

colors = {
    'inact_dist_same': 'navy',
    'act_dist_same': 'darkgreen',
    'inact_dist_diff': 'cornflowerblue',
    'act_dist_diff': 'mediumseagreen',
    'same': 'darkorange',
    'diff': 'indigo'
}
name_dict = {
    'inact_dist_same': 'Inactive/Same',
    'act_dist_same': 'Active/Same',
    'inact_dist_diff': 'Inactive/Different',
    'act_dist_diff': 'Active/Different',
    'same': 'Same',
    'diff': 'Diff'
}


def init_args():
    parser = argparse.ArgumentParser(description='AllReID tracker')
    parser.add_argument('--dir_name', type=str,
                        default='features',
                        help='Input feature path')
    parser.add_argument('--save_dir', type=str,
                        default='histograms_features',
                        help='Output feature path')
    parser.add_argument('--plot_seq_plots', type=bool, default=True)
    parser.add_argument('--inact', type=bool, default=True)
    parser.add_argument('--only_sd', type=bool, default=False)
    return parser.parse_args()


def get_feat(id_feature_dict, gt_id, proxy, mv_avg):
    until = min(60, len(id_feature_dict[gt_id]))
    if proxy == 'last' or (
            len(id_feature_dict[gt_id]) == 1 and proxy != 'mv_avg' and proxy != 'each' and proxy != 'eachmedian'):
        feat = id_feature_dict[gt_id][-1]
    elif proxy == 'mean':
        feat = np.mean(np.stack(id_feature_dict[gt_id][-until:]), axis=0)
    elif proxy == 'median':
        feat = np.median(np.stack(id_feature_dict[gt_id][-until:]), axis=0)
    elif proxy == 'mode':
        feat = scipy.stats.mode(
            np.stack(id_feature_dict[gt_id][-until:]), axis=0)[0][0]
    elif proxy == 'each' or proxy == 'eachmedian':
        if len(id_feature_dict[gt_id]) == 1:
            feat = [id_feature_dict[gt_id]]
        else:
            feat = id_feature_dict[gt_id]
    elif proxy == 'mv_avg':
        if gt_id not in mv_avg.keys():
            feat = mv_avg[gt_id] = id_feature_dict[gt_id][-1]
        else:
            feat = mv_avg[gt_id] = 0.9 * mv_avg[gt_id] + \
                (1-0.9) * id_feature_dict[gt_id][-1]
    return feat


def emd(c, d):
    # compute earth movers distance
    return scipy.stats.wasserstein_distance(c, d)


def get_intersection(v_1, v_2, v_o_1, v_o_2,
                     threshs=[l/1000 for l in list(range(0, 1000, 5))]):
    # intersection new with 0.5 quantille
    intersection = list()
    for i in range(0, 129):
        intersection.append(np.abs(v_1[i]-v_2[i]))

    auc = np.array([sum(intersection[:i])/sum(intersection)
                   for i in range(len(intersection))])
    idx = list()
    for thresh in threshs:
        idx.append(np.where(auc > thresh)[0][0]/100)

    thresh = find_thresh(v_o_1, v_o_2, thresh_inc=0.01)
    thresh = round(thresh*100)/100

    return idx, thresh


def cost(thresh, diff_weight, same_id_dist, diff_id_dist):
    return (1-diff_weight) * (100 - stats.percentileofscore(same_id_dist, thresh)) + \
        diff_weight * (stats.percentileofscore(diff_id_dist, thresh))


def find_thresh(same_id_dist, diff_id_dist, thresh_inc=0.1, diff_weight=0.5):
    """
    Given a threshold X:
    - the % of misclassified SAME ID distances (i.e. False Negatives) is % of SAME ID distances ABOVE the thresold (= 1 -stats.percentileofscore(same_id_dist, X))
    - the % of misclassified DIFF ID distances (i.e. False Positives) is % of DIFF ID distances BELOW the thresold (= stats.percentileofscore(diff_id_dist, X))
    - An ideal threshold should minimize the sum of FNs+FPs. Therefore we minimize:
       100 -stats.percentileofscore(same_id_dist, X) + stats.percentileofscore(diff_id_dist, X)

    - Note 1: this is the discretized version of minimizing the area unde the curve of the distributions
    of positive (resp. negative) distances that are right (resp. left) of the threshold (i.e. probabilities of FNs / FPs) 
    - Note 2: it might be that we actually need to weight FPs and FNs differently!?
    """

    _min_thresh = np.median(same_id_dist)
    _max_thresh = np.median(diff_id_dist)
    min_thresh = min(_min_thresh, _max_thresh)
    max_thresh = max(_min_thresh, _max_thresh)
    thresholds = np.arange(min_thresh, max_thresh, thresh_inc)

    value_per_cost = {}
    for thresh in thresholds:
        value_per_cost[thresh] = \
            cost(thresh, diff_weight, same_id_dist, diff_id_dist)

    min_cost_thresh = min(value_per_cost, key=lambda x: value_per_cost[x])
    return min_cost_thresh


def main():
    args = init_args()

    paths = os.listdir(args.dir_name)
    os.makedirs(args.save_dir, exist_ok=True)
    bins_for_hist = np.arange(0, 1.3, 0.01)
    patience = 50
    proxys = ['mean', 'mv_avg', 'each', 'median', 'mode', 'last', 'eachmedian']

    for proxy in proxys:
        for i, path in enumerate(paths):

            splitted = path.split('_')

            # get proxy method you want to use
            if splitted[0] == 'first':
                BB = splitted[3]
                dets = splitted[1]
            else:
                BB = splitted[2]
                dets = splitted[0]

            logger.info(f'PROXY {proxy}, dir {args.dir_name}, PATH {path}')
            save_name = '_'.join([dets, BB, proxy, str(patience)])
            logger.info(save_name)

            # make dir to store
            os.makedirs(osp.join(args.save_dir, save_name), exist_ok=True)

            # load data
            with open(osp.join(args.dir_name, path)) as json_file:
                data = json.load(json_file)

            # set variable keys
            if not args.only_sd:
                kps = [
                    'inact_dist_same',
                    'act_dist_same',
                    'inact_dist_diff',
                    'act_dist_diff']
            else:
                kps = [
                    'same',
                    'diff']

            # initlalize seq plot
            if args.plot_seq_plots:
                num_seqs = len(data)
                width = num_seqs / 2 * 13
                fig = plt.figure(figsize=(width, 20), dpi=100)

                plt.rcParams.update({
                    'font.size': 30,
                    'legend.fontsize': 30,
                    'axes.labelsize': 30,
                    'axes.titlesize': 30,
                    'xtick.labelsize': 30,
                    'ytick.labelsize': 30})

            # initalize storage
            dists_all = defaultdict(list)
            seq_dict = defaultdict(dict)

            intersection_quant_act = list()
            quantilles_act = list()
            quantilles_inact = list()
            intersection_quant_inact = list()
            intersection_guillem_inact = list()
            intersection_guillem_act = list()

            emd_act = list()
            emd_inact = list()

            # ITERATE OVER SEQUECES
            seq_list = list(data.keys()) + ['Overall']
            for i, (k, seq) in enumerate(data.items()):
                id_feature_dict = defaultdict(list)
                id_time_dict = defaultdict(list)
                mv_avg = dict()
                dists = defaultdict(list)
                values_hist_ = dict()
                values_orig_ = dict()
                active_ids = list()

                if args.plot_seq_plots:
                    fig.add_subplot(2, math.ceil(num_seqs/2), i+1)

                # iterate over frames
                for frame, detections in seq.items():
                    frame = int(frame)
                    active_ids_new = list()
                    to_add_feats = dict()
                    to_add_time = dict()
                    gt_ids = list()
                    # iterate over detections
                    for det in detections:
                        gt_ids.append(det['gt_id'])
                        # ignore bounding boxes that were not matched to any gt bb
                        if det['gt_id'] == -1:
                            continue
                        # get current features
                        curr_feat = np.array(det['feats'])

                        # if we already have features for current detection
                        if det['gt_id'] in id_feature_dict.keys():
                            print(det['gt_id'], det['gt_id'] in active_ids, det['gt_id'] in list(id_feature_dict.keys()))
                            # for active take last detection to get dist and add to dist dict
                            if det['gt_id'] in active_ids:
                                last_feat = id_feature_dict[det['gt_id']][-1]
                                d = distance.cosine(last_feat, curr_feat)
                                dists['act_dist_same'].append(d)

                            # for inactive compute proxy distance and add to dist dict
                            else:
                                if len(id_time_dict[det['gt_id']]):
                                    if frame - id_time_dict[diff][-1] > patience:
                                        continue
                                last_feat = get_feat(
                                    id_feature_dict, det['gt_id'], proxy, mv_avg)
                                if proxy == 'each':
                                    dist = [distance.cosine(
                                        last, curr_feat) for last in last_feat]
                                    d = sum(dist)/len(dist)
                                elif proxy == 'eachmedian':
                                    dist = np.array(
                                        [distance.cosine(last, curr_feat) for last in last_feat])
                                    d = np.median(dist)
                                else:
                                    d = distance.cosine(last_feat, curr_feat)
                                dists['inact_dist_same'].append(d)

                        # get distances to tracks of different classes
                        for diff in id_feature_dict.keys():
                            if diff == det['gt_id']:
                                continue

                            # if in active tracks take last detection
                            if diff in active_ids:
                                diff_feat = id_feature_dict[diff][-1]
                                d = distance.cosine(diff_feat, curr_feat)
                                dists['act_dist_diff'].append(d)
                            # if in inactive tracks compute proxy
                            else:
                                if len(id_time_dict[diff]):
                                    if frame - id_time_dict[diff][-1] > patience:
                                        continue
                                last_feat = get_feat(
                                    id_feature_dict, diff, proxy, mv_avg)
                                if proxy == 'each':
                                    dist = [distance.cosine(
                                        last, curr_feat) for last in last_feat]
                                    d = sum(dist)/len(dist)
                                elif proxy == 'eachmedian':
                                    dist = np.array(
                                        [distance.cosine(last, curr_feat) for last in last_feat])
                                    d = np.median(dist)
                                else:
                                    d = distance.cosine(last_feat, curr_feat)
                                dists['inact_dist_diff'].append(d)

                        # update active ids --> new detections ids are active tracks now
                        active_ids_new.append(det['gt_id'])
                        to_add_feats[det['gt_id']] = curr_feat
                        to_add_time[det['gt_id']] = frame

                    for key in to_add_feats.keys():
                        id_feature_dict[key].append(to_add_feats[key])
                        id_time_dict[key].append(to_add_time[key])

                    active_ids = active_ids_new

                # plot sequence histograms
                if args.plot_seq_plots:
                    if args.only_sd:
                        dists['same'] = dists['inact_dist_same'] + \
                            dists['act_dist_same']
                        dists['diff'] = dists['inact_dist_diff'] + \
                            dists['act_dist_diff']
                    for kp in kps:
                        v = dists[kp]
                        dists_all[kp].extend(v)
                        kpv = np.array(v)
                        seq_dict[k][kp] = kpv
                        # density=True --> converts histogram in PDF
                        n, _, _ = plt.hist(
                            kpv, bins_for_hist, facecolor=colors[kp], alpha=0.75, density=True)
                        values_hist_[kp] = n
                        values_orig_[kp] = kpv

                # SEQUENCE: compute intersection points and quantilles / using qualtilles
                # between active tracks and detections of same and different classes
                if args.only_sd:
                    act_same = values_hist_['same']
                    act_diff = values_hist_['diff']
                    v_act_same = values_orig_['same']
                    v_act_diff = values_orig_['diff']
                else:
                    act_same = values_hist_['act_dist_same']
                    act_diff = values_hist_['act_dist_diff']
                    v_act_same = values_orig_['act_dist_same']
                    v_act_diff = values_orig_['act_dist_diff']

                quant, thresh = get_intersection(
                    act_diff, act_same, v_act_same, v_act_diff)
                plt.scatter(quant[100], 0, marker='*', color='red', s=400)
                plt.scatter(thresh, 0, marker='*', color='blue', s=400)
                intersection_quant_act.append(quant[100])
                intersection_guillem_act.append(thresh)
                quantilles_act.append(quant)

                # SEQUENCE: compute emd between active tracks and detections of same and different class histograms
                emd_act.append(emd(v_act_same, v_act_diff))

                # SEQUENCE: compute intersection points and quantilles / using qualtilles
                # between inactive tracks and detections of same and different classes
                if args.inact and not args.only_sd:
                    inact_same = values_hist_['inact_dist_same']
                    inact_diff = values_hist_['inact_dist_diff']
                    v_inact_same = values_orig_['inact_dist_same']
                    v_inact_diff = values_orig_['inact_dist_diff']

                    if len(v_inact_same):
                        quant, thresh = get_intersection(
                            inact_diff, inact_same, v_inact_same, v_inact_diff)
                    else:
                        quant, thresh = get_intersection(
                            inact_diff, np.zeros(inact_same.shape), [10.0], v_inact_diff)

                    plt.scatter(quant[100], 0, marker='^', color='red', s=400)
                    plt.scatter(thresh, 0, marker='^', color='blue', s=400)
                    intersection_quant_inact.append(quant[100])
                    intersection_guillem_inact.append(thresh)
                    quantilles_inact.append(quant)

                    # SEQUENCE: compute emd between inactive tracks and detections of same and different class histograms
                    if len(v_inact_same):
                        emd_inact.append(emd(v_inact_same, v_inact_diff))
                    else:
                        emd_inact.append(emd([10.0], v_inact_diff))

                    # legend for plots
                    leg = ['intersect act', 'intersect inact',] + [name_dict[k] for k in kps]
                else:
                    # legend for plots
                    leg = ['intersect act'] + [name_dict[k] for k in kps]

                # add legend and title to sequence plot
                if args.plot_seq_plots:
                    plt.legend(leg, loc='upper left')
                    plt.xlabel('Cosine Distance', fontweight="bold")
                    plt.ylim(bottom=-0.2, top=7)
                    k = k.split('_')[0]
                    k1 = ''.join([c for c in k if not c.isdigit()])
                    k2 = ''.join([c for c in k if c.isdigit()])
                    if len(k2) < 2:
                        k2 = '0' + k2
                    plt.title(' '.join([k1, k2]), fontweight="bold")

            # add overall plot to sequence plots
            if args.plot_seq_plots:
                fig.add_subplot(2, math.ceil(num_seqs/2), i+2)
                values_hist_ = dict()
                for j, kp in enumerate(kps):
                    dists = dists_all[kp]
                    kpv = np.array(dists)
                    n, _, _ = plt.hist(kpv, bins_for_hist, facecolor=colors[
                        kp], alpha=0.75, density=True)
                    values_hist_[kp] = n
                    values_orig_[kp] = kpv

                # OVERALL: compute intersection points and quantilles / using qualtilles
                # between active tracks and detections of same and different classes
                if args.only_sd:
                    act_same = values_hist_['same']
                    act_diff = values_hist_['diff']
                    v_act_same = values_orig_['same']
                    v_act_diff = values_orig_['diff']
                else:
                    act_same = values_hist_['act_dist_same']
                    act_diff = values_hist_['act_dist_diff']
                    v_act_same = values_orig_['act_dist_same']
                    v_act_diff = values_orig_['act_dist_diff']

                quant, thresh = get_intersection(
                    act_diff, act_same, v_act_same, v_act_diff)
                quants = [quant[100]]
                plt.scatter(quant[100], 0, marker='*', color='red', s=400)
                plt.scatter(thresh, 0, marker='*', color='blue', s=400)
                intersection_quant_act.append(quant[100])
                intersection_guillem_act.append(thresh)
                quantilles_act.append(quant)

                # OVERALL: compute emd between active tracks and detections of same and different class histograms
                emd_act.append(emd(v_act_same, v_act_diff))

                # OVERALL: compute intersection points and quantilles / using qualtilles
                # between inactive tracks and detections of same and different classes
                if args.inact and not args.only_sd:
                    inact_diff = values_hist_['inact_dist_diff']
                    inact_same = values_hist_['inact_dist_same']
                    v_inact_same = values_orig_['inact_dist_same']
                    v_inact_diff = values_orig_['inact_dist_diff']
                    
                    if len(v_inact_same):
                        quant, thresh = get_intersection(
                            inact_diff, inact_same, v_inact_same, v_inact_diff)
                    else:
                        quant, thresh = get_intersection(
                            inact_diff, np.zeros(inact_same.shape), [10.0], v_inact_diff)
                    
                    quants.append(quant[100])
                    plt.scatter(quant[100], 0, marker='^', color='red', s=400)
                    plt.scatter(thresh, 0, marker='^', color='blue', s=400)
                    intersection_quant_inact.append(quant[100])
                    intersection_guillem_inact.append(thresh)
                    quantilles_inact.append(quant)

                    # OVERALL: compute emd between inactive tracks and detections of same and different class histograms
                    if len(v_inact_same):
                        emd_inact.append(emd(v_inact_same, v_inact_diff))
                    else:
                        emd_inact.append(emd([10.0], v_inact_diff))

                # legends for overall plot
                if args.only_sd:
                    plt.legend(['intersect'] +
                            [name_dict[k] for k in kps], loc='upper left')
                else:
                    plt.legend(['intersect act', 'intersect inact'] + 
                            [name_dict[k] for k in kps], loc='upper left')
                plt.xlabel('Cosine Distance', fontweight="bold")
                plt.ylim(bottom=-0.2, top=7)
                plt.title("Overall", fontweight="bold")

                # save sequence + overall plot
                if "bdd" not in path:
                    plt.tight_layout()
                    add = 'only_sd' if args.only_sd else ""
                    plt.savefig(osp.join(args.save_dir,  save_name,
                                add + 'Sequences' + path + '.png'))
                plt.close()

            '''# Compute EMD BETWEEN SEQS histograms
            emd_between = defaultdict(list)
            add = 'only_sd' if args.only_sd else ""
            for _, kpv1 in seq_dict.items():
                l1, l2, l3, l4 = list(), list(), list(), list()
                list_ = [l1, l2, l3, l4]
                for _, kpv2 in seq_dict.items():
                    for i, k in enumerate(kpv1.keys()):
                        list_[i].append(emd(kpv1[k], kpv2[k]))
                for i, k in enumerate(kpv1.keys()):
                    emd_between[k].append(list_[i])'''

            logger.info("inter act")
            logger.info(intersection_quant_act)
            logger.info(intersection_guillem_act)
            logger.info("inter inact")
            logger.info(intersection_quant_inact)
            logger.info(intersection_guillem_inact)

            # GENERATE EMD PER SEQENCE PLOT --> EMD between histogram of active/
            # inactive & detections of same and different class per seq
            import matplotlib

            fig = plt.figure()
            ax = fig.add_subplot(111)

            plt.plot(list(range(len(emd_act))), emd_act, color=colors['same'])
            rect1 = matplotlib.patches.Rectangle((0, min(emd_act)), len(emd_act), max(
                emd_act)-min(emd_act), alpha=0.1, facecolor=colors['same'])
            ax.add_patch(rect1)
            plt.hlines(min(emd_act), 0, len(emd_act),
                    color=colors['same'], linestyles='--', alpha=0.5)
            plt.hlines(max(emd_act), 0, len(emd_act),
                    color=colors['same'], linestyles='--', alpha=0.5)

            if not args.only_sd:
                plt.plot(list(range(len(emd_act))),
                        emd_inact, color=colors['diff'])
                rect2 = matplotlib.patches.Rectangle((0, min(emd_inact)), len(emd_act), max(
                    emd_inact)-min(emd_inact), alpha=0.1, facecolor=colors['diff'])
                ax.add_patch(rect2)
                plt.hlines(min(emd_inact), 0, len(emd_inact),
                        color=colors['diff'], linestyles='--', alpha=0.5)
                plt.hlines(max(emd_inact), 0, len(emd_inact),
                        color=colors['diff'], linestyles='--', alpha=0.5)

            plt.xticks(list(range(len(emd_act))),
                    seq_list, rotation=45, fontsize=12)
            plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], [
                    0.2, 0.4, 0.6, 0.8, 1.0], fontsize=12)
            plt.title('EMD', fontsize=12)

            if not args.only_sd:
                plt.legend(['active', 'inactive'], fontsize=12)
            else:
                plt.legend(['emd'], fontsize=12)

            plt.tight_layout()
            add = 'only_sd' if args.only_sd else ''
            plt.savefig(osp.join(args.save_dir, save_name,
                        add + 'EMD_' + path + '.png'))
            plt.close()

            # GENERATE INTERSECTION POINTS PER SEQENCE PLOT --> INTERSECTION POINTS
            # between histogram of active/inactive & detections of same and different
            # class per seq
            fig = plt.figure()
            ax = fig.add_subplot(111)

            plt.plot(
                list(range(len(intersection_quant_act))),
                intersection_quant_act,
                color=colors['same'])
            rect1 = matplotlib.patches.Rectangle(
                (0, min(intersection_quant_act)),
                len(intersection_quant_inact),
                max(intersection_quant_act)-min(intersection_quant_act),
                alpha=0.1,
                facecolor=colors['same'])
            ax.add_patch(rect1)
            plt.hlines(
                min(intersection_quant_act),
                0,
                len(intersection_quant_inact),
                color=colors['same'],
                linestyles='--',
                alpha=0.5)
            plt.hlines(
                max(intersection_quant_act),
                0,
                len(intersection_quant_inact),
                color=colors['same'],
                linestyles='--',
                alpha=0.5)

            if not args.only_sd:
                plt.plot(
                    list(range(len(intersection_quant_inact))),
                    intersection_quant_inact, color=colors['diff'])
                rect2 = matplotlib.patches.Rectangle(
                    (0, min(intersection_quant_inact)),
                    len(intersection_quant_inact),
                    max(intersection_quant_inact)-min(intersection_quant_inact),
                    alpha=0.1,
                    facecolor=colors['diff'])
                ax.add_patch(rect2)
                plt.hlines(
                    min(intersection_quant_inact),
                    0,
                    len(intersection_quant_inact),
                    color=colors['diff'],
                    linestyles='--',
                    alpha=0.5)
                plt.hlines(
                    max(intersection_quant_inact),
                    0,
                    len(intersection_quant_inact),
                    color=colors['diff'],
                    linestyles='--',
                    alpha=0.5)

            plt.xticks(
                list(range(len(intersection_quant_inact))),
                seq_list,
                rotation=45,
                fontsize=12)
            plt.yticks(
                [0.2, 0.4, 0.6, 0.8, 1.0],
                [0.2, 0.4, 0.6, 0.8, 1.0], fontsize=12)
            plt.title('Intersection points', fontsize=12)

            if not args.only_sd:
                plt.legend(['active', 'inactive'], fontsize=12)
            else:
                plt.legend(['intersection points'], fontsize=12)

            plt.tight_layout()
            add = 'only_sd' if args.only_sd else ''
            plt.savefig(osp.join(args.save_dir, save_name, add +
                        'Intersection_points_' + path + '.png'))
            plt.close()

            # GENERATE QUANTILLES PLOT FOR ACTIVE TRACKS
            fig = plt.figure(figsize=[8, 4.5])
            ax = fig.add_subplot(111)
            for quantille in quantilles_act:
                plt.plot(quantille, list(range(len(quantille))))
            plt.xticks([0.2, 0.4, 0.6, 0.8, 1.0], [
                    0.2, 0.4, 0.6, 0.8, 1.0], fontsize=12)
            plt.yticks([40, 80, 120, 160, 200], [
                    0.2, 0.4, 0.6, 0.8, 1.0], fontsize=12)
            plt.legend(seq_list, fontsize=12, loc='lower right')
            add = 'only_sd' if args.only_sd else ''
            plt.savefig(osp.join(args.save_dir, save_name, add +
                        'quantilles_act' + path + '.png'))
            plt.close()

            # GENERATE QUANTILLES PLOT FOR INACTIVE TRACKS
            if not args.only_sd:
                fig = plt.figure(figsize=[8, 4.5])
                ax = fig.add_subplot(111)
                for quantille in quantilles_inact:
                    plt.plot(quantille, list(range(len(quantille))))
                plt.xticks([0.2, 0.4, 0.6, 0.8, 1.0], [
                        0.2, 0.4, 0.6, 0.8, 1.0], fontsize=12)
                plt.yticks([40, 80, 120, 160, 200], [
                        0.2, 0.4, 0.6, 0.8, 1.0], fontsize=12)
                plt.legend(seq_list, fontsize=12, loc='lower right')
                add = 'only_sd' if args.only_sd else ''
                plt.savefig(osp.join(args.save_dir, save_name, add +
                            'quantilles_inact' + path + '.png'))
                plt.close()

            # GENERATE OVERALL PLOT ONLY
            fig = plt.figure(figsize=(9, 9), dpi=100)
            plt.rcParams.update({'font.size': 42})
            kps = list()

            # get names for plot
            if not args.only_sd:
                _kps = [
                    'inact_dist_same',
                    'act_dist_same',
                    'inact_dist_diff',
                    'act_dist_diff']
            else:
                _kps = [
                    'same',
                    'diff'
                ]
            bins_ = dict()
            plt.scatter(quants[0], 0, marker='*', color='red', s=800, zorder=1)
            if not args.only_sd:
                plt.scatter(quants[1], 0, marker='v', color='red', s=800, zorder=1)
            for i, kp in enumerate(_kps):
                dists = dists_all[kp]
                kps.append(kp)
                kpv = np.array(dists)
                # density=True --> converts histogram in PDF
                # bc of IoU distance often == 1.0
                kpv = np.delete(kpv, np.where(kpv == 1.0))
                n, _, _ = plt.hist(kpv, 100, density=True, facecolor=colors[
                    kp], alpha=0.75, zorder=-1)
                bins_[kp] = n

            if args.only_sd:
                plt.legend(['intersect'] + [name_dict[k] for k in kps], loc='upper right', fontsize=26)
            else:
                plt.legend(['intersect act', 'intersect inact'] + [name_dict[k] for k in kps], loc='upper right', fontsize=26)
            plt.xlabel('IoU Distance', fontweight="bold")
            plt.title('Over all Sequences', fontweight="bold")
            plt.grid(True, color='0.95', linestyle='-', linewidth=1)
            plt.xticks(rotation=30)
            plt.ylim(bottom=-0.2, top=7)
            plt.tight_layout()
            add = 'only_sd' if args.only_sd else ""
            plt.savefig(osp.join(args.save_dir,  save_name, add + path + '.png'))
            plt.close()

if __name__ == "__main__":
    main()