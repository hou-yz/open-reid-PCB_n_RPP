from __future__ import print_function, absolute_import
import os.path as osp

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json

import pandas as pd

class vehicleID(object):
    def __init__(self, root, type='reid', fps=2, trainval=False):
        if type == 'tracking_gt':
            train_file = '~/Data/VehicleID_V1.0/train_test_split/train_list.txt'
            val_file = '~/Data/VehicleID_V1.0/train_test_split/test_list_800.txt'
            train_dir = '~/Data/VehicleID_V1.0/image/'
            val_dir = '~/Data/VehicleID_V1.0/image/'
            self.train_file = osp.expanduser(train_file)
            self.gallery_file = osp.expanduser(val_file)
            self.query_file = osp.expanduser(val_file)
            self.train_path = osp.expanduser(train_dir)
            self.gallery_path = osp.expanduser(val_dir)
            self.query_path = osp.expanduser(val_dir)

        self.train, self.query, self.gallery = [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids = 0, 0, 0

        self.type = type
        self.load()

    def preprocess(self, file, path, type='reid'):
        with open(file) as f:
            image_list = f.readlines()
        image_list = [item.strip('\n') for item in image_list]
        all_pids = {}
        ret = []
        for item in image_list:
            item_list = item.split(' ')
            image_num = item_list[0]
            pid = int(item_list[1])
            fname = image_num + '.jpg'
            cam = 0
            ret.append((fname, pid, cam))

            if pid not in all_pids:
                all_pids[pid] = pid
        return ret, int(len(all_pids))

    def load(self):
        self.train, self.num_train_ids = self.preprocess(self.train_file, self.train_path, self.type)
        self.gallery, self.num_gallery_ids = self.preprocess(self.gallery_file, self.gallery_path, self.type)
        self.query, self.num_query_ids = self.preprocess(self.query_file, self.query_path, self.type)

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:5d} | {:8d}"
              .format(self.num_train_ids, len(self.train)))
        print("  query    | {:5d} | {:8d}"
              .format(self.num_query_ids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}"
              .format(self.num_gallery_ids, len(self.gallery)))


