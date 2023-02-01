import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict
from scipy import stats
import os.path as osp
import os


as_ = ['split_1', 'split_2', 'split_3'] #'each_sample_evalBB1' #'each_sample_weighted_motion' #'each_sample_evalBB1'
load_dir = 'distances' # 'distances/new_distances'
save_dir = osp.join('histograms', 'splits')

trackers = os.listdir(os.path.join(load_dir, 'split_1'))
trackers = [t.split('_')[0] for t in trackers]
colors = ['teal', 'indigo', 'darkorange', 'darkred', 'green', 'saddlebrown']
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

plot_seq_plots = True

def get_mean_var_std(bins, n):
    mids = 0.5*(bins[1:] + bins[:-1])
    mean = np.average(mids, weights=n)
    var = np.average((mids - mean)**2, weights=n)
    return mean, var, np.sqrt(var)

os.makedirs(save_dir, exist_ok=True)
all_act = dict()
all_inact = dict()
for tracker in trackers:
    dists_all = defaultdict(list)
    kps = [
        'inact_dist_same',
        'act_dist_same',
        'inact_dist_diff',
        'act_dist_diff']
    for a in as_:
        for path in os.listdir(osp.join(load_dir, a)):
            if tracker == path.split('_')[0]:
                with open(osp.join(load_dir, a, path)) as json_file:
                    data = json.load(json_file)
                print(a, path, tracker)
                os.makedirs(osp.join(save_dir, tracker), exist_ok=True)

                for i, (k, seq) in enumerate(data.items()):
                    for j, kp in enumerate(kps):
                        # dists_all[kp.split('_')[-1]].extend(seq[kp])
                        dists_all[kp].extend(seq[kp])

    fig = plt.figure(figsize=(9, 9), dpi=100)
    plt.rcParams.update({'font.size': 42})
    kps = list()
    a = 0
    b = 0 
    c = 0
    d = 0
    e = 0
    for i, (kp, dists) in enumerate(dists_all.items()):
        # if 'inact'  in kp:
        #    continue
        kps.append(kp)
        kpv = np.array(dists)
        # kpv = np.delete(kpv, np.where(kpv==1))
        # kpv = np.where(np.isnan(kpv), 1.2, kpv)
        if kp == 'inact_dist_same':
            a = kpv.mean()
            b = kpv.std()
        if kp == 'act_dist_diff':
            c = kpv.mean()
        if kp == 'same':
            d = kpv.std()
        if kp == 'diff':
            e = kpv.mean()
        # print(kp, kpv.mean(), kpv.std(), stats.mode(kpv)[0][0], np.median(kpv), kpv.min(), kpv.max())
        # density=True --> converts histogram in PDF
        kpv = np.delete(kpv, np.where(kpv ==1.0))
        n, bins, patches = plt.hist(kpv, 100, density=True, facecolor=colors[
            kp], alpha=0.75)

        ### GET MEAN AND STD
        mean, var, std = get_mean_var_std(bins, n)
        # print('Overall', kp, mean, var, std)

        # plt.axvline(x=kpv.mean(), color=colors[kp], linestyle='--', linewidth=3)
        # c = lambda x: np.round(x, decimals=2)
        # text = "{}, {}, {}, {}".format(c(kpv.std()), c(kpv.mean()), c(
        #     stats.mode(kpv)[0][0]), c(np.median(kpv)))
        # print(text)
        # plt.text(bins[np.argmax(n)]-0.1, np.max(n), text, va='center', fontsize=5)
    print('inactive', a-0.5*b, 'active', c-b, 'only one', e-d)
    all_act[tracker] = round((c-b)*1000)/1000
    all_inact[tracker] = round((a-0.5*b)*1000)/1000
    print()
    # plt.ylim(0, 8)
    plt.legend([name_dict[k] for k in kps], loc='upper right', fontsize=26)
    plt.xlabel('IoU Distance', fontweight="bold")
    plt.title('STUFF', fontweight="bold")
    plt.grid(True, color='0.95', linestyle='-', linewidth=1)
    plt.xticks(rotation=30)
    # plt.ylim(top=4.7)
    # plt.xlim(0, 1.2)
    plt.tight_layout()
    plt.savefig(osp.join(save_dir,  tracker, path + '.png'))
    plt.close()

det_files = [
            "qdtrack.txt",
            "CenterTrack.txt",
            "CSTrack.txt",
            "FairMOT.txt",
            "JDE.txt",
            "TraDeS.txt",
            "TransTrack.txt",
            "CenterTrackPub.txt",
            "center_track.txt",
            "tracktor_prepr_det.txt"]
order = [d.split('.')[0].split('_')[0] for d in det_files]

print(all_act)
print(all_inact)
print([all_act[t] for t in order])
print([all_inact[t] for t in order])