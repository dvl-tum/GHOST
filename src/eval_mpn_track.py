import motmetrics as mm
import os.path as osp
from collections import OrderedDict
from pathlib import Path
import logging


logger = logging.getLogger('AllReIDTracker.EvalMPNTrack')


def get_results(
        names,
        corr_gt_as_gt=None,
        gt_as_track_res=None,
        dataset_cfg=None,
        dir=None,
        tracker=None):
    # set default solver for evaluation
    mm.lap.default_solver = 'lapsolver'

    # normal GT files or reduced gt files
    if corr_gt_as_gt:
        gt = corr_gt_as_gt
    else:
        gt_path = osp.join(dataset_cfg['mot_dir'], dir)
        gtfiles = [osp.join(gt_path, i, 'gt/gt.txt') for i in names]
        gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f,
                         fmt='mot15-2D', min_confidence=1)) for f in gtfiles])

    # GT as track results or normal out files
    if gt_as_track_res:
        ts = gt_as_track_res
    else:
        out_mot_files_path = osp.join('out', tracker.experiment)
        tsfiles = [osp.join(out_mot_files_path, '%s' % i) for i in names]
        ts = OrderedDict([(osp.splitext(Path(f).parts[-1])[0],
                         mm.io.loadtxt(f, fmt='mot15-2D')) for f in tsfiles])

    # if validation set 50-50 is used, use only second half for evaluation
    if dataset_cfg['validation_set_gt']:
        for k, v in gt.items():
            import numpy as np
            min_frame = np.min(ts[k].index.get_level_values('FrameId').values)
            mask = v.index.get_level_values('FrameId').values >= min_frame
            gt[k] = v[mask]

    accs, names = compare_dataframes(gt, ts)

    return accs, names


def compare_dataframes(gts, ts):
    """Builds accumulator for each sequence."""
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            accs.append(mm.utils.compare_to_groundtruth(
                gt=gts[k],
                dt=tsacc,
                dist='iou',
                distth=0.5))
            names.append(k)

    return accs, names


def get_summary(accs, names, name):
    logger.info(name)
    mh = mm.metrics.create()
    metrics = mm.metrics.motchallenge_metrics + \
        ['num_objects', 'idtp', 'idfn', 'idfp', 'num_predictions']
    summary = mh.compute_many(
        accs,
        names=names,
        metrics=metrics,
        generate_overall=True)

    logger.info(mm.io.render_summary(summary, formatters=mh.formatters,
                                     namemap=mm.io.motchallenge_metric_names))
