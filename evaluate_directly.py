import os
import os.path as osp
from src.eval_track_eval import *
from TrackEval import trackeval
import logging

logger = logging.getLogger('AllReIDTracker')
logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(message)s')
formatter = logging.Formatter('%(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(ch)

ddir = 'split_finals' #'split_hs_search' #'split_wo_motion' #'split_diff_thresh' # 'split'
trackers = os.listdir(osp.join('out', ddir))
trackers = [osp.join(ddir, t) for t in trackers]
cfg = {'mot_dir': '/storage/slurm/seidensc/datasets/MOT/MOT17', 'detector': 'FRCNN'}

def evaluate(dir, tracker, dataset_cfg, log=True):
    eval_config, dataset_config, metrics_config = setup_trackeval()
    gt_path = osp.join(dataset_cfg['mot_dir'], dir)
    dataset_config['GT_FOLDER'] = gt_path
    dataset_config['TRACKERS_FOLDER'] = 'out'
    dataset_config['TRACKERS_TO_EVAL'] = [tracker]
    dataset_config['OUTPUT_FOLDER'] = 'track_eval_output'
    dataset_config['PRINT_CONFIG'] = False
    eval_config['PRINT_CONFIG'] = False
    dataset_config['SEQ_INFO'] = get_dict(dataset_cfg['mot_dir'], dataset_cfg['detector'])
    dataset_config['SKIP_SPLIT_FOL'] = True
    dataset_config['TRACKER_SUB_FOLDER'] = ''
    eval_config['DISPLAY_LESS_PROGRESS'] = False
    eval_config['TIME_PROGRESS'] = False
    metrics_config['PRINT_CONFIG'] = False

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    if not log:
        evaluator.config['PRINT_RESULTS'] = False
    print(evaluator.config['PRINT_RESULTS'])
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]

    metrics_list = []
    for metric in [
            trackeval.metrics.HOTA,
            trackeval.metrics.CLEAR,
            trackeval.metrics.Identity]:
        # trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)

    return output_res, output_msg


for tracker in trackers:
    output_res, _ = evaluate('train', tracker, cfg)
