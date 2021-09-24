from pathlib import Path
import motmetrics as mm
from collections import OrderedDict
import os
import os.path as osp
from shutil import copyfile


class Tester():
    def __init__(self, dataset_cfg, dire='train'):
        self.dataset_cfg = dataset_cfg
        self.dir = dire
        self.fake_tracktor()

    def fake_tracktor(self):
        names = os.listdir(os.path.join(self.dataset_cfg['det_dir'], self.dir))
        os.makedirs('out_tracktor', exist_ok=True)
        track_files = os.path.join(self.dataset_cfg['det_dir'], self.dir)
        tsfiles = [os.path.join(track_files, '%s' % i, 'det/tracktor_prepr_det.txt') for i in names]
        tsfiles_new = list()
        for ts, name in zip(tsfiles, names):
            new_name = osp.join('out_tracktor', name)
            copyfile(ts, new_name)
            tsfiles_new.append(new_name)
        self.tsfiles = tsfiles_new

    def evaluate(self):
        names = os.listdir(os.path.join(self.dataset_cfg['det_dir'], self.dir))
        accs, names = self._get_results(names)
        mh = mm.metrics.create()
        metrics = mm.metrics.motchallenge_metrics + ['num_objects', 'idtp',
                        'idfn', 'idfp', 'num_predictions']

        summary = mh.compute_many(accs, names=names,
                                metrics=metrics,
                                generate_overall=True)

        print(mm.io.render_summary(summary, formatters=mh.formatters,
                                    namemap=mm.io.motchallenge_metric_names))

        return None, None

    def _get_results(self, names):
        mm.lap.default_solver = 'lapsolver'
        gt_path = osp.join(self.dataset_cfg['mot_dir'], self.dir)
        gtfiles = [os.path.join(gt_path, i, 'gt/gt.txt') for i in names]
        #track_files = os.path.join(self.dataset_cfg['det_dir'], self.dir)
        #tsfiles = [os.path.join(track_files, '%s' % i, 'det/tracktor_prepr_det.txt') for i in names]
        tsfiles = self.tsfiles
        gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D',
                            min_confidence=1)) for f in gtfiles])
        ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f,
                            fmt='mot15-2D')) for f in tsfiles])

        accs, names = self._compare_dataframes(gt, ts)

        return accs, names

    @staticmethod
    def _compare_dataframes(gts, ts):
        """Builds accumulator for each sequence."""
        accs = []
        names = []
        for k, tsacc in ts.items():
            if k in gts:
                accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc,
                                    'iou', distth=0.5))
                names.append(k)

        return accs, names

if __name__ == "__main__":
    dataset_cfg = {'mot_dir': "../../datasets/MOT/MOT17",
            'det_dir': "../../datasets/MOT/MOT17Labels",
            'det_file': "tracktor_prepr_det.txt"}
    tester = Tester(dataset_cfg)
    tester.evaluate()
