from TrackEval import trackeval
import argparse
import os.path as osp



def get_dict(mot_dir, detectors):
    if "MOT17" in mot_dir and detectors != 'all':
        seq_length = {
            'MOT17-02-FRCNN': None, 'MOT17-04-FRCNN': None,
            'MOT17-05-FRCNN': None, 'MOT17-09-FRCNN': None,
            'MOT17-10-FRCNN': None, 'MOT17-11-FRCNN': None,
            'MOT17-13-FRCNN': None}
    elif "MOT17" in mot_dir and detectors == 'all':
        seq_length = {
            'MOT17-02-FRCNN': None, 'MOT17-04-FRCNN': None,
            'MOT17-05-FRCNN': None, 'MOT17-09-FRCNN': None,
            'MOT17-10-FRCNN': None, 'MOT17-11-FRCNN': None,
            'MOT17-13-FRCNN': None, 
            'MOT17-02-DPM': None, 'MOT17-04-DPM': None,
            'MOT17-05-DPM': None, 'MOT17-09-DPM': None,
            'MOT17-10-DPM': None, 'MOT17-11-DPM': None,
            'MOT17-13-DPM': None, 
            'MOT17-02-SDP': None, 'MOT17-04-SDP': None,
            'MOT17-05-SDP': None, 'MOT17-09-SDP': None,
            'MOT17-10-SDP': None, 'MOT17-11-SDP': None,
            'MOT17-13-SDP': None}
    elif "MOT" in mot_dir:
        seq_length = {
            'MOT20-01': None, 'MOT20-02': None,
            'MOT20-03': None, 'MOT20-05': None}
    else:
        seq_length = {}
    # seq_length = {
    #         'MOT17-13-FRCNN': None}
    return seq_length


def setup_trackeval():
    # default eval config
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False

    # default dataset config
    default_dataset_config = \
        trackeval.datasets.BDD100K.get_default_dataset_config()

    # default metrics config
    default_metrics_config = {
        'METRICS': [
            'HOTA',
            'CLEAR',
            'Identity'],
        'THRESHOLD': 0.5}

    config = {**default_eval_config, **default_dataset_config,
              **default_metrics_config}  # Merge default configs

    # generate config argument parser
    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if isinstance(
                config[setting],
                list) or isinstance(
                config[setting],
                type(None)):
            parser.add_argument("--" + setting, nargs='+')
        else:
            parser.add_argument("--" + setting)

    # update config dict with args from argument parser
    args = parser.parse_args().__dict__
    for setting in args.keys():
        if args[setting] is not None:
            if isinstance(config[setting], type(True)):
                if args[setting] == 'True':
                    x = True
                elif args[setting] == 'False':
                    x = False
                else:
                    raise Exception(
                        'Command line parameter ' +
                        setting +
                        'must be True or False')
            elif isinstance(config[setting], type(1)):
                x = int(args[setting])
            elif isinstance(args[setting], type(None)):
                x = None
            elif setting == 'SEQ_INFO':
                x = dict(zip(args[setting], [None] * len(args[setting])))
            else:
                x = args[setting]
            config[setting] = x
    
    # get updated config dicts
    eval_config = {
        k: v for k,
        v in config.items() if k in default_eval_config.keys()}
    dataset_config = {
        k: v for k,
        v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {
        k: v for k,
        v in config.items() if k in default_metrics_config.keys()}

    return eval_config, dataset_config, metrics_config


def evaluate_track_eval_bdd(dir, tracker, dataset_cfg, log=True):
    import os
    eval_config, dataset_config, metrics_config = setup_trackeval()
    mot_dir = os.path.dirname(dataset_cfg['mot_dir'])
    gt_path = osp.join(mot_dir, 'labels', 'box_track_20', dir)
    dataset_config['GT_FOLDER'] = gt_path
    dataset_config['TRACKERS_FOLDER'] = 'out'
    dataset_config['TRACKERS_TO_EVAL'] = [tracker.experiment]
    dataset_config['OUTPUT_FOLDER'] = 'track_eval_output'
    dataset_config['PRINT_CONFIG'] = False
    eval_config['PRINT_CONFIG'] = False
    # dataset_config['SEQ_INFO'] = get_dict(dataset_cfg['mot_dir'], dataset_cfg['detector'])
    dataset_config['SKIP_SPLIT_FOL'] = True
    dataset_config['TRACKER_SUB_FOLDER'] = ''
    eval_config['DISPLAY_LESS_PROGRESS'] = False
    eval_config['TIME_PROGRESS'] = False
    metrics_config['PRINT_CONFIG'] = False

    if 'debug' in dataset_cfg['splits']:
        seq_list = ['b23c9e00-b425de1b']
    else:
        seq_list = None

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    if not log:
        evaluator.config['PRINT_RESULTS'] = False
    dataset_list = [trackeval.datasets.BDD100K(dataset_config, seq_list=seq_list)]

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


def just_evaluate_track_eval_bdd(dir='val', log=True):
    experiment_list = [
        'bdd100k_0.851840_evalBB:0_each_sample2:0.75:last_frame:0.8MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.5',
        'bdd100k_0.851840_evalBB:0_each_sample2:0.8:last_frame:0.8MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.5',
        'bdd100k_0.851840_evalBB:0_each_sample2:0.85:last_frame:0.8MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.5',
        'bdd100k_cls_score_0.851840_evalBB:0_each_sample2:0.7:last_frame:0.7MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.7',
        'bdd100k_cls_score_0.851840_evalBB:0_each_sample2:0.7:last_frame:0.75MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.7',
        'bdd100k_cls_score_0.851840_evalBB:0_each_sample2:0.7:last_frame:0.8MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.7',
        'bdd100k_cls_score_0.851840_evalBB:0_each_sample2:0.7:last_frame:0.85MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.7',
        'bdd100k_cls_score_0.851840_evalBB:0_each_sample2:0.7:last_frame:0.85MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.6',
        'bdd100k_cls_score_0.851840_evalBB:0_each_sample2:0.7:last_frame:0.8MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.6',
        'bdd100k_cls_score_0.851840_evalBB:0_each_sample2:0.7:last_frame:0.75MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.6',
        'bdd100k_cls_score_0.851840_evalBB:0_each_sample2:0.7:last_frame:0.7MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.6',
        'bdd100k_cls_score_0.851840_evalBB:0_each_sample2:0.7:last_frame:0.7MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.5',
        'bdd100k_cls_score_0.851840_evalBB:0_each_sample2:0.7:last_frame:0.75MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.5',
        'bdd100k_cls_score_0.851840_evalBB:0_each_sample2:0.7:last_frame:0.8MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.5',
        'bdd100k_cls_score_0.851840_evalBB:0_each_sample2:0.7:last_frame:0.85MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.5'
    ]
    experiment_list = [
        'bdd100k_0.851840_evalBB:0_each_sample2:0.85:last_frame:0.75MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.6',
        'bdd100k_0.851840_evalBB:0_each_sample2:0.8:last_frame:0.75MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.6',
        'bdd100k_0.851840_evalBB:0_each_sample2:0.75:last_frame:0.75MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.6',
        'bdd100k_0.851840_evalBB:0_each_sample2:0.65:last_frame:0.75MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.6'
    ]
    experiment_list = [
        'bdd100k_cls_score_0.851840_evalBB:0_each_sample2:0.8:last_frame:0.75MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.5',
        'bdd100k_cls_score_0.851840_evalBB:0_each_sample2:0.75:last_frame:0.75MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.6',
        'bdd100k_cls_score_0.851840_evalBB:0_each_sample2:0.65:last_frame:0.75MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.6',
        'bdd100k_cls_score_0.851840_evalBB:0_each_sample2:0.85:last_frame:0.75MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.5',
        'bdd100k_cls_score_0.851840_evalBB:0_each_sample2:0.85:last_frame:0.75MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.6',
        'bdd100k_cls_score_0.851840_evalBB:0_each_sample2:0.8:last_frame:0.75MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.6',
        'bdd100k_cls_score_0.851840_evalBB:0_each_sample2:0.65:last_frame:0.75MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.5',
        'bdd100k_cls_score_0.851840_evalBB:0_each_sample2:0.75:last_frame:0.75MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.5',
        'bdd100k_cls_score_0.851840_evalBB:0_each_sample2:0.65:last_frame:0.75MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.6',
    ]
    experiment_list = [
        'bdd100k_0.851840_evalBB:0_each_sample2:0.8000001:last_frame:0.75MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.5'
    ]
    experiment_list = [
        'qdtrack_0.851840_evalBB:0_each_sample2:0.7:last_frame:0.75MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10.0',
        'qdtrack_0.851840_evalBB:0_each_sample2:0.7:last_frame:0.8MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10.0',
        'qdtrack_0.851840_evalBB:0_each_sample2:0.7:last_frame:0.85MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10.0',
        'qdtrack_0.851840_evalBB:0_each_sample2:0.7:last_frame:0.7MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10.0'
    ]

    experiment_list = [
        'bdd100k_0.851840_evalBB:0_each_sample2:0.7:last_frame:0.7MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10',
        'bdd100k_0.851840_evalBB:0_each_sample2:0.7:last_frame:0.75MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10',
        'bdd100k_0.851840_evalBB:0_each_sample2:0.7:last_frame:0.8MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10',
        'bdd100k_0.851840_evalBB:0_each_sample2:0.7:last_frame:0.85MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10',
        'bdd100k_0.851840_evalBB:0_each_sample2:0.70000005:last_frame:0.7MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10',
        'bdd100k_0.851840_evalBB:0_each_sample2:0.70000005:last_frame:0.75MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10',
        'bdd100k_0.851840_evalBB:0_each_sample2:0.70000005:last_frame:0.8MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10',
        'bdd100k_0.851840_evalBB:0_each_sample2:0.70000005:last_frame:0.85MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10',
        'bdd100k_0.851840_evalBB:0_each_sample2:0.70000001:last_frame:0.85MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10',
        'bdd100k_0.851840_evalBB:0_each_sample2:0.70000001:last_frame:0.8MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10',
        'bdd100k_0.851840_evalBB:0_each_sample2:0.70000001:last_frame:0.75MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10',
        'bdd100k_0.851840_evalBB:0_each_sample2:0.70000001:last_frame:0.7MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10',
        'bdd100k_0.851840_evalBB:0_each_sample2:0.650000001:last_frame:0.75MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10',
        'bdd100k_0.851840_evalBB:0_each_sample2:0.750000001:last_frame:0.75MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10',
        'bdd100k_0.851840_evalBB:0_each_sample2:0.80000001:last_frame:0.75MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10',
        'bdd100k_0.851840_evalBB:0_each_sample2:0.850000001:last_frame:0.75MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10',
        'bdd100k_0.851840_evalBB:0_each_sample2:0.75:last_frame:0.8MM:1sum0.30.30.3InactPat:10000000ConfThresh:0.5'
    ]

    experiment_list = [
        'bdd100k_0.851840_evalBB:0_each_sample2:0.65:last_frame:0.8MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10',
        'bdd100k_0.851840_evalBB:0_each_sample2:0.75:last_frame:0.8MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10',
        'bdd100k_0.851840_evalBB:0_each_sample2:0.8:last_frame:0.8MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10',
        'bdd100k_0.851840_evalBB:0_each_sample2:0.85:last_frame:0.8MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10',
        'bdd100k_0.851840_evalBB:0_each_sample2:0.650000005:last_frame:0.8MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10',
        'bdd100k_0.851840_evalBB:0_each_sample2:0.750000005:last_frame:0.8MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10',
        'bdd100k_0.851840_evalBB:0_each_sample2:0.80000005:last_frame:0.8MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10',
        'bdd100k_0.851840_evalBB:0_each_sample2:0.850000005:last_frame:0.8MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10'
    ]

    experiment_list = [
        'bdd100k_0.851840_evalBB:0_each_sample2:0.850000005:last_frame:0.8MM:1sum0.30.30.3InactPat:10000000ConfThresh:-10'
    ]

    seq_list = None
    
    mot_dir = '/storage/slurm/seidensc/datasets/BDD100/bdd100k'
    import os
    eval_config, dataset_config, metrics_config = setup_trackeval()
    gt_path = osp.join(mot_dir, 'labels', 'box_track_20', dir)
    dataset_config['GT_FOLDER'] = gt_path
    dataset_config['TRACKERS_FOLDER'] = 'out'
    dataset_config['TRACKERS_TO_EVAL'] = experiment_list
    dataset_config['OUTPUT_FOLDER'] = 'track_eval_output'
    dataset_config['PRINT_CONFIG'] = False
    eval_config['PRINT_CONFIG'] = False
    # dataset_config['SEQ_INFO'] = get_dict(dataset_cfg['mot_dir'], dataset_cfg['detector'])
    dataset_config['SKIP_SPLIT_FOL'] = True
    dataset_config['TRACKER_SUB_FOLDER'] = ''
    eval_config['DISPLAY_LESS_PROGRESS'] = False
    eval_config['TIME_PROGRESS'] = False
    metrics_config['PRINT_CONFIG'] = False

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    if not log:
        evaluator.config['PRINT_RESULTS'] = False
    dataset_list = [trackeval.datasets.BDD100K(dataset_config, seq_list=seq_list)]

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
