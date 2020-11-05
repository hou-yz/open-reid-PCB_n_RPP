from __future__ import print_function, absolute_import

import os.path as osp
import re
from glob import glob


class Market1501(object):
    def __init__(self, root):
        """
        Initialize the ims.

        Args:
            self: (todo): write your description
            root: (str): write your description
        """
        self.root = osp.join(root, 'Market-1501-v15.09.15')
        self.train_path = osp.join(root, 'Market-1501-v15.09.15/bounding_box_train')
        self.gallery_path = osp.join(root, 'Market-1501-v15.09.15/bounding_box_test')
        self.query_path = osp.join(root, 'Market-1501-v15.09.15/query')
        self.camstyle_path = osp.join(root, 'Market-1501-v15.09.15/bounding_box_train_camstyle')
        self.train, self.query, self.gallery, self.camstyle = [], [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids, self.num_camstyle_ids = 0, 0, 0, 0
        self.num_cams = 6
        self.load()

    def preprocess(self, path, relabel=True):
        """
        Preprocess a list of files.

        Args:
            self: (todo): write your description
            path: (str): write your description
            relabel: (todo): write your description
        """
        pattern = re.compile(r'([-\d]+)_c(\d+)')
        all_pids = {}
        ret = []
        fpaths = sorted(glob(osp.join(path, '*.jpg')))
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            ret.append((fname, pid, cam - 1))
        return ret, int(len(all_pids))

    def load(self):
        """
        Loads the corpus.

        Args:
            self: (todo): write your description
        """
        self.train, self.num_train_ids = self.preprocess(self.train_path)
        self.gallery, self.num_gallery_ids = self.preprocess(self.gallery_path, False)
        self.query, self.num_query_ids = self.preprocess(self.query_path, False)
        self.camstyle, self.num_camstyle_ids = self.preprocess(self.camstyle_path)

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:5d} | {:8d}"
              .format(self.num_train_ids, len(self.train)))
        print("  query    | {:5d} | {:8d}"
              .format(self.num_query_ids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}"
              .format(self.num_gallery_ids, len(self.gallery)))
        print("  camstyle  | {:5d} | {:8d}"
              .format(self.num_camstyle_ids, len(self.camstyle)))
