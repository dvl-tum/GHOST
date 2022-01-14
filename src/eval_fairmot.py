import os
import numpy as np
import copy
import motmetrics as mm
import os
from typing import Dict
import numpy as np
import logging


logger = logging.getLogger('AllReIDTracker.EvalFairMOT')
logger.propagate = False

mm.lap.default_solver = 'lap'


class Evaluator(object):

    def __init__(self, data_root, seq_name, data_type):
        self.data_root = data_root
        self.seq_name = seq_name
        self.data_type = data_type

        self.load_annotations()
        self.reset_accumulator()

    def load_annotations(self):
        assert self.data_type == 'mot'

        gt_filename = os.path.join(
            self.data_root, self.seq_name, 'gt', 'gt.txt')
        self.gt_frame_dict = read_results(
            gt_filename, self.data_type, is_gt=True)
        self.gt_ignore_frame_dict = read_results(
            gt_filename, self.data_type, is_ignore=True)

    def reset_accumulator(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids, rtn_events=False):
        # results
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)

        # gts
        gt_objs = self.gt_frame_dict.get(frame_id, [])
        gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]

        # ignore boxes
        ignore_objs = self.gt_ignore_frame_dict.get(frame_id, [])
        ignore_tlwhs = unzip_objs(ignore_objs)[0]

        # indicator if valid or ignore
        # num_gt = len(gt_objs)
        # num_ignore = len(ignore_objs)
        # ndicator = np.array([0]*num_gt+[1]*num_ignore, dtype=bool)

        # stack gt and ignore to match both at the same time
        # ignore_tlwhs = np.vstack([gt_tlwhs, ignore_tlwhs])

        # remove ignored results
        keep = np.ones(len(trk_tlwhs), dtype=bool)
        iou_distance = mm.distances.iou_matrix(
            ignore_tlwhs, trk_tlwhs, max_iou=0.5)
        if len(iou_distance) > 0:
            match_is, match_js = mm.lap.linear_sum_assignment(iou_distance)
            match_is, match_js = map(
                lambda a: np.asarray(
                    a, dtype=int), [
                    match_is, match_js])
            match_ious = iou_distance[match_is, match_js]

            match_js = np.asarray(match_js, dtype=int)
            match_js = match_js[np.logical_not(np.isnan(match_ious))]

            # mask if matched to gt and not ignore
            # match_js = match_js[indicator[match_is]]

            keep[match_js] = False
            trk_tlwhs = trk_tlwhs[keep]
            trk_ids = trk_ids[keep]

        # get distance matrix
        iou_distance = mm.distances.iou_matrix(
            gt_tlwhs, trk_tlwhs, max_iou=0.5)

        # acc
        self.acc.update(gt_ids, trk_ids, iou_distance)

        if rtn_events and iou_distance.size > 0 and hasattr(
                self.acc, 'last_mot_events'):
            # only supported by https://github.com/longcw/py-motmetrics
            events = self.acc.last_mot_events
        else:
            events = None
        return events

    def eval_file(self, filename):
        self.reset_accumulator()

        result_frame_dict = read_results(filename, self.data_type, is_gt=False)

        frames = sorted(list(set(result_frame_dict.keys())))
        # frames = sorted(list(range(frames[0], frames[-1])))

        print('Frames are consecutive', self.is_consecutive(frames), filename)
        for frame_id in frames:
            trk_objs = result_frame_dict.get(frame_id, [])
            trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
            # frame_id, bb coordinates, person id
            self.eval_frame(frame_id, trk_tlwhs, trk_ids, rtn_events=False)

        return self.acc

    @staticmethod
    def is_consecutive(input_list):
        return input_list[-1]-input_list[0] == len(input_list)-1

    @staticmethod
    def get_summary(
        accs,
        names,
        metrics=(
            'mota',
            'num_switches',
            'idp',
            'idr',
            'idf1',
            'precision',
            'recall')):
        names = copy.deepcopy(names)
        if metrics is None:
            metrics = mm.metrics.motchallenge_metrics
        metrics = copy.deepcopy(metrics)

        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs,
            metrics=metrics,
            names=names,
            generate_overall=True
        )

        return summary

    @staticmethod
    def save_summary(summary, filename):
        import pandas as pd
        writer = pd.ExcelWriter(filename)
        summary.to_excel(writer)
        writer.save()


def write_results(filename, results_dict: Dict, data_type: str):
    if not filename:
        return
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.makedirs(path)

    if data_type in ('mot', 'mcmot', 'lab'):
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian -1 -1 -10 {x1} {y1} {x2} {y2} -1 -1 -1 -1000 -1000 -1000 -10 {score}\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, frame_data in results_dict.items():
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in frame_data:
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(
                    frame=frame_id,
                    id=track_id,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    w=w,
                    h=h,
                    score=1.0)
                f.write(line)
    print('Save results to {}'.format(filename))


def read_results(filename, data_type: str, is_gt=False, is_ignore=False):
    if data_type in ('mot', 'lab'):
        read_fun = read_mot_results
    else:
        raise ValueError('Unknown data type: {}'.format(data_type))

    return read_fun(filename, is_gt, is_ignore)


"""
labels={'ped', ...			% 1
'person_on_vhcl', ...	% 2
'car', ...				% 3
'bicycle', ...			% 4
'mbike', ...			% 5
'non_mot_vhcl', ...		% 6
'static_person', ...	% 7
'distractor', ...		% 8
'occluder', ...			% 9
'occluder_on_grnd', ...		%10
'occluder_full', ...		% 11
'reflection', ...		% 12
'crowd' ...			% 13
};
"""


def read_mot_results(filename, is_gt, is_ignore):
    '''
    returns: frame dict with list of bbs in given frame
    if is_gt = True: only valid_labels are taken into account
    if is_ignore = True: only invalid_labels are taken into account
    '''
    valid_labels = {1}
    ignore_labels = {2, 7, 8, 12}
    results_dict = dict()

    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            for line in f.readlines():
                linelist = line.split(',')
                if len(linelist) < 7:
                    print('cont 1')
                    continue
                fid = int(linelist[0])
                if fid < 1:
                    print('cont 2')
                    continue
                results_dict.setdefault(fid, list())

                # only get valid gt bbs
                if is_gt:
                    if 'MOT16-' in filename or 'MOT17-' in filename:
                        label = int(float(linelist[7]))
                        mark = int(float(linelist[6]))
                        if mark == 0 or label not in valid_labels:
                            continue
                    score = 1

                # only get invalid gt bbs
                elif is_ignore:
                    if 'MOT16-' in filename or 'MOT17-' in filename:
                        label = int(float(linelist[7]))
                        vis_ratio = float(linelist[8])
                        if label not in ignore_labels and vis_ratio >= 0:
                            continue
                    else:
                        continue
                    score = 1
                else:
                    score = float(linelist[6])

                # box_size = float(linelist[4]) * float(linelist[5])
                # if box_size > 7000:
                # if box_size <= 7000 or box_size >= 15000:
                # if box_size < 15000:
                    # continue

                tlwh = tuple(map(float, linelist[2:6]))
                target_id = int(linelist[1])

                results_dict[fid].append((tlwh, target_id, score))

    return results_dict


def unzip_objs(objs):
    if len(objs) > 0:
        tlwhs, ids, scores = zip(*objs)
    else:
        tlwhs, ids, scores = [], [], []
    tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)

    return tlwhs, ids, scores
