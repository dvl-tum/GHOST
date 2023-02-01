import collections
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict
from scipy import stats
import os.path as osp
import os

kps_names = [
        'inact_dist_same',
        'act_dist_same',
        'inact_dist_diff',
        'act_dist_diff']


def get_binned_dict(default=list):
    binned_dict = {
        0: defaultdict(default),
        0.1: defaultdict(default),
        0.2: defaultdict(default),
        0.3: defaultdict(default),
        0.4: defaultdict(default),
        0.5: defaultdict(default),
        0.6: defaultdict(default),
        0.7: defaultdict(default),
        0.8: defaultdict(default),
        0.9: defaultdict(default)}

    return binned_dict


def inst_dict():
    dists = get_binned_dict()
    for k, v in dists.items():
        for k1 in ['inact_dist_same', 'inact_dist_diff', 'act_dist_same', 'act_dist_diff', 'dist_diff']:
            v[k1] = list()

    return dists


def get_frame_data(seq, frame):
    inter = np.array(seq['interaction_mat'][frame])
    occ = np.array(seq['occlusion_mat'][frame])
    act = np.array(seq['active_inactive'][frame])
    same_class = np.array(seq['same_class_mat'][frame])
    dist = np.array(seq['dist'][frame])
    height = np.array(seq['size'][frame])
    # iou_dist = np.array(seq['iou_dist'][frame])
    # inactive_count = np.array(seq['inactive_count'][frame])

    return inter, occ, act, same_class, dist, height #, iou_dist, inactive_count


def normal_hist_fig(data, name, save_dir, path, limit=True, num_bins=100):
    fig = plt.figure(figsize=(52, 20), dpi=100)

    data_all = list()
    for i, (k, seq) in enumerate(data.items()):
        fig.add_subplot(2, 4, i+1)

        data_all.extend(seq)

        kpv = np.array(seq)
        n, bins, patches = plt.hist(
            kpv, num_bins, facecolor=colors[0], density=True, alpha=0.75)

        plt.xlabel(name)
        plt.ylabel('Counts')
        #plt.ylim(bottom=-0.2, top=25)
        if limit:
            plt.xlim(0, 1.0)
        plt.title(k)

    fig.add_subplot(2, 4, i+2)
    kpv = np.array(data_all)
    n, bins, patches = plt.hist(
        kpv, num_bins, facecolor=colors[0], density=True, alpha=0.75)

    plt.xlabel(name)
    plt.ylabel('Counts')
    # plt.ylim(bottom=-0.2, top=25)
    if limit:
        plt.xlim(0, 1.0)
    plt.title('Overall')
    plt.savefig(osp.join(save_dir,  path[:-5], name + '_histogram_' + path + '.png'))
    plt.close()


def get_hist_fig(path, param, dists, save_dir):
    fig = plt.figure(figsize=(65, 20), dpi=100)
    plt.rcParams.update({'font.size': 22})
    stats_dict = defaultdict(dict)
    for i, (occ, dist_dict) in enumerate(dists.items()):
        # fig = plt.figure(figsize=(15, 13), dpi=100)
        fig.add_subplot(2, 5, i+1)
        kps = list()
        # for j, (kp, _dists) in enumerate(dist_dict.items()):
        for j, kp in enumerate(kps_names):
            _dists = dist_dict[kp]
            if kp == 'dist_diff':
                continue

            if i == 0:
                stats_dict[kp]['mean'] = list()
                stats_dict[kp]['std'] = list()

            kps.append(kp)

            if not len(_dists):
                stats_dict[kp]['mean'].append(0)
                stats_dict[kp]['std'].append(0)
                continue

            kpv = np.array(_dists)
            # print(kp, kpv.mean(), kpv.std(), stats.mode(kpv)[0][0], \
            #     np.median(kpv), kpv.min(), kpv.max())
            # density=True --> converts histogram in PDF
            n, bins, patches = plt.hist(
                kpv, 100, density=True, facecolor=colors[j], alpha=0.75)

            stats_dict[kp]['mean'].append(kpv.mean())
            stats_dict[kp]['std'].append(kpv.std())

        plt.ylim(bottom=-0.2, top=25)
        plt.legend(kps)
        plt.xlabel('Cosine Distance')
        plt.ylabel('Counts')
        plt.title(occ)
        plt.xlim(0, 1.0)

    plt.savefig(osp.join(save_dir, path[:-5], param + path + '.png'))
    plt.close()

    return stats_dict


def get_stats_plot(stats_dict, path, param, save_dir):
    linestyle = ['-', '--']
    plt.rcParams.update({'font.size': 12})
    for s in ['mean', 'std']:
        for i, (kp, kp_stats) in enumerate(stats_dict.items()):
            std = np.array(kp_stats['std'])
            for j, (stat, stat_values) in enumerate(kp_stats.items()):
                if stat == s:
                    x = np.arange(0, 1.0, 0.1)
                    y = np.array(stat_values)
                    x = x[std != 0]
                    y = y[std != 0]
                    plt.plot(x, y, linestyle=linestyle[j], c=colors[i])
        plt.legend(list(stats_dict.keys()))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig(osp.join(save_dir, path[:-5], param + s + path + '.png'))
    plt.close()


def plot_dist_diff(dists, path, param, save_dir):
    fig = plt.figure(figsize=(65, 20), dpi=100)
    plt.rcParams.update({'font.size': 22})
    for i, (occ, dist_dict) in enumerate(dists.items()):
        # fig = plt.figure(figsize=(15, 13), dpi=100)
        fig.add_subplot(2, 5, i+1)
        kps = list()
        for j, (kp, _dists) in enumerate(dist_dict.items()):
            if kp != 'dist_diff':
                continue
            kps.append(kp)

            if not len(_dists):
                continue

            kpv = np.array(_dists)
            # print(kp, kpv.mean(), kpv.std(), stats.mode(kpv)[0][0], \
            #     np.median(kpv))
            # density=True --> converts histogram in PDF
            n, bins, patches = plt.hist(
                kpv, 100, facecolor=colors[j], density=True, alpha=0.75)

        # plt.ylim(0, 8)
        plt.legend(kps)
        plt.xlabel('Cosine Distance')
        plt.ylabel('Counts')
        plt.title(occ)
        # plt.xlim(0, 1.0)

    plt.savefig(osp.join(save_dir, path[:-5], param + '_diff' + path + '.png'))
    plt.close()


def analyse(data, save_dir, path, colors, take_max=True):

    for param in ['occlusion', 'interaction']:
        max_param = 0
        dists = inst_dict()
        occlusions = dict()
        interactions = dict()
        heights = dict()
        iou_dist = dict()
        for k, seq in data.items():
            occlusions[k] = list()
            interactions[k] = list()
            heights[k] = list()
            iou_dist[k] = defaultdict(dict)
            for frame in range(len(seq['interaction_mat'])):
                inter, occ, act, same_class, dist, height = get_frame_data(seq, frame)
                occlusions[k].extend(np.sum(occ, axis=1).tolist())
                interactions[k].extend(np.sum(inter, axis=1).tolist())
                heights[k].extend(height.tolist())
                
                for samp in range(dist.shape[0]):
                    # get occlusion/interaction of given sample &
                    # corresponding occluder/interacters in current frame
                    if param == 'occlusion':
                        if take_max:
                            # take only dist to max occlusion --> :-4 = dummy
                            # boxes for out of frame bounding boxes
                            # o, idx = np.max(occ[samp, :-4]), [np.argmax(occ[samp][:-4])]

                            # take all occluders into account
                            o = np.sum(occ[samp])
                            idx = np.nonzero(occ[samp, :-4])[0]

                        else:
                            # take dist to all interactors into account
                            o = np.sum(occ[samp])
                            idx = np.nonzero(occ[samp])[0]

                    elif param == 'interaction':
                        o = np.sum(inter[samp]) / 1
                        idx = np.nonzero(inter[samp])[0]

                    max_param = o if o > max_param else max_param

                    # get previous sample of occluded object if present
                    occluded = np.nonzero(same_class[samp])[0][0] if \
                        np.nonzero(same_class[samp])[0].shape[0] > 0 else None

                    # round values leq than 1 to 0.9999 for binning reasons
                    if o >= 1.0:
                        o = 0.9999
                    o = float(np.floor(o*10)/10)

                    # get distance between previous samp and samp of curr obj
                    a, b, min_dist_diff = None, None, 100
                    if occluded is not None:
                        b = dist[samp, occluded]
                        if occluded in np.nonzero(act[0, :])[0]:
                            dists[o]['act_dist_same'].extend([b])
                        else:
                            dists[o]['inact_dist_same'].extend([b])

                    for i in idx:
                        # get prev samps corr to interacting/occluding samps
                        occluding = np.nonzero(same_class[i])[0][0] if \
                            np.nonzero(same_class[i])[0].shape[0] > 0 else None

                        # get distance to samp
                        if occluding is not None:
                            a = dist[samp, occluding]
                            if occluding in np.nonzero(act[0, :])[0]:
                                dists[o]['act_dist_diff'].extend([a])
                            else:
                                dists[o]['inact_dist_diff'].extend([a])

                        # update distance difference
                        if a and b:
                            min_dist_diff = a-b if a-b < min_dist_diff else min_dist_diff

                    # check how similar distance is. Best case: a-b = 1.0
                    if a and b:
                        dists[o]['dist_diff'].extend([min_dist_diff])

            print('Max', param, max_param)        

        # hist fig
        stats_dict = get_hist_fig(path, param, dists, save_dir)

        # hist fig occlusion and interaction
        normal_hist_fig(occlusions, 'occlusions', save_dir, path, num_bins=10)
        normal_hist_fig(interactions, 'interactions', save_dir, path, num_bins=10)
        normal_hist_fig(heights, 'height', save_dir, path, limit=False)

        # plot stats
        get_stats_plot(stats_dict, path, param, save_dir)

        # plot dist difference
        plot_dist_diff(dists, path, param, save_dir)


def main(load_dir, save_dir, paths, colors):

    for path in paths:
        print(path)
        if path != 'center_track_0.851840_evalBB:1_last_frame:0.5:last_frame:0.7InactPat:1000000distances.json' and path != 'center_track_0.851840_evalBB:1_each_sample2:0.55:last_frame:0.7InactPat:1000000distances.json':
            continue

        # make save dir and load file
        os.makedirs(osp.join(save_dir, path[:-5]), exist_ok=True)
        with open(osp.join(load_dir, path)) as json_file:
            data = json.load(json_file)

        analyse(data, save_dir, path, colors)


if __name__ == "__main__":
    a = 'new_distances' #'each_sample_evalBB1'
    load_dir = 'distances/' + a # 'distances/new_distances'
    save_dir = osp.join('histograms', a)

    paths = os.listdir(load_dir)
    colors = ['teal', 'indigo', 'darkorange', 'darkred', 'green', 'saddlebrown']

    main(load_dir, save_dir, paths, colors)
