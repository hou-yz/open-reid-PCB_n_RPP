from __future__ import print_function, absolute_import

import os.path as osp
import re
from glob import glob


class DukeMTMC(object):

    def __init__(self, root, data_type='reid', iCams=None, fps=1, trainval=False):
        """
        Initialize training data.

        Args:
            self: (todo): write your description
            root: (str): write your description
            data_type: (str): write your description
            iCams: (todo): write your description
            fps: (todo): write your description
            trainval: (todo): write your description
        """
        if iCams is None:
            iCams = list(range(1, 9))
        if data_type == 'tracking_gt':
            self.root = osp.join(root, 'DukeMTMC')
            if not trainval:
                train_dir = osp.join(root, 'DukeMTMC/ALL_gt_bbox/train')
            else:
                train_dir = osp.join(root, 'DukeMTMC/ALL_gt_bbox/trainval')
            val_dir = osp.join(root, 'DukeMTMC/ALL_gt_bbox/val')
            self.train_path = osp.join(train_dir, f'gt_bbox_{fps}_fps')
            self.gallery_path = osp.join(val_dir, 'gt_bbox_1_fps')
            self.query_path = osp.join(val_dir, 'gt_bbox_1_fps')
        elif data_type == 'tracking_det':
            self.root = root
            self.train_path = root
            self.gallery_path = None
            self.query_path = None
        elif data_type == 'reid':  # reid
            self.root = osp.join(root, 'DukeMTMC-reID')
            self.train_path = osp.join(root, 'DukeMTMC-reID/bounding_box_train')
            self.gallery_path = osp.join(root, 'DukeMTMC-reID/bounding_box_test')
            self.query_path = osp.join(root, 'DukeMTMC-reID/query')
        else:
            raise Exception

        self.camstyle_path = osp.join(root, 'DukeMTMC-reID/bounding_box_train_camstyle')
        self.train, self.query, self.gallery, self.camstyle = [], [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids, self.num_camstyle_ids = 0, 0, 0, 0
        self.num_cams = 8

        self.data_type = data_type
        self.iCams = iCams
        self.load()

    def preprocess(self, path, relabel=True, type='reid'):
        """
        Preprocesses a list of pids.

        Args:
            self: (todo): write your description
            path: (str): write your description
            relabel: (todo): write your description
            type: (str): write your description
        """
        if type == 'tracking_det':
            pattern = re.compile(r'c(\d+)_f(\d+)')
        else:
            pattern = re.compile(r'([-\d]+)_c(\d+)')
        all_pids = {}
        ret = []
        if path is None:
            return ret, int(len(all_pids))
        if type == 'tracking_gt':
            fpaths = []
            for iCam in self.iCams:
                fpaths += sorted(glob(osp.join(path, 'camera' + str(iCam), '*.jpg')))
        else:
            fpaths = sorted(glob(osp.join(path, '*.jpg')))
        for fpath in fpaths:
            fname = osp.basename(fpath)
            if type == 'tracking_det':
                cam, frame = map(int, pattern.search(fname).groups())
                pid = 8000
            else:
                pid, cam = map(int, pattern.search(fname).groups())
            if type == 'tracking_gt':
                fname = osp.join('camera' + str(cam), osp.basename(fpath))
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
        Loads training process.

        Args:
            self: (todo): write your description
        """
        self.train, self.num_train_ids = self.preprocess(self.train_path, True, self.data_type)
        self.gallery, self.num_gallery_ids = self.preprocess(self.gallery_path, False, self.data_type)
        self.query, self.num_query_ids = self.preprocess(self.query_path, False, self.data_type)
        self.camstyle, self.num_camstyle_ids = self.preprocess(self.camstyle_path, True, self.data_type)

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
