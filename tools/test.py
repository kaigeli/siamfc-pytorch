from __future__ import absolute_import

import os
from got10k.experiments import *

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    net_path = 'pretrained/siamfc_alexnet_e50.pth'
    tracker = TrackerSiamFC(net_path=net_path)
    root_dir = os.path.expanduser('~/2_dataset/OTB100')
    e = ExperimentOTB(root_dir, version=2015)
    # 2013: __otb13_seqs,
    # 2015: __tb100_seqs,
    # 'otb2013': __otb13_seqs,
    # 'otb2015': __tb100_seqs,
    # 'tb50': __tb50_seqs,
    # 'tb100': __tb100_seqs
    # e = ExperimentOTB(root_dir, version='tb50')
    # e = ExperimentUAV123(root_dir, version='tb50') to do
    e.run(tracker)
    e.report([tracker.name])

