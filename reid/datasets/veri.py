from __future__ import print_function, absolute_import
import os.path as osp

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json

import re
import hashlib
import shutil
from glob import glob
from zipfile import ZipFile
import xml.etree.ElementTree as et


class VeRi(object):
    def __init__(self, root, type='reid', fps=2, trainval=False):
        if type == 'tracking_gt':
            train_dir = '~/Data/VeRi/image_train/'
            val_dir = '~/Data/VeRi/image_query/'
            self.train_path = osp.expanduser(train_dir)
            self.gallery_path = osp.expanduser(val_dir)
            self.query_path = osp.expanduser(val_dir)

        self.train, self.query, self.gallery = [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids = 0, 0, 0

        self.type = type
        self.load()

    def preprocess(self, path, type='reid'):
        pattern = re.compile(r'(\d+)_c(\d+)')
        all_pids = {}
        ret = []
        fpaths = sorted(glob(osp.join(path, '*.jpg')))
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue
            if pid not in all_pids:
                all_pids[pid] = len(all_pids)
            pid = all_pids[pid]
            ret.append((fname, pid, cam))
        return ret, int(len(all_pids))

    def load(self):
        self.train, self.num_train_ids = self.preprocess(self.train_path, self.type)
        self.gallery, self.num_gallery_ids = self.preprocess(self.gallery_path, self.type)
        self.query, self.num_query_ids = self.preprocess(self.query_path, self.type)

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:5d} | {:8d}"
              .format(self.num_train_ids, len(self.train)))
        print("  query    | {:5d} | {:8d}"
              .format(self.num_query_ids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}"
              .format(self.num_gallery_ids, len(self.gallery)))