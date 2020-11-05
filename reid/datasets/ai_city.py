from __future__ import print_function, absolute_import

import os.path as osp
import re
import xml.dom.minidom as XD
from collections import defaultdict
from glob import glob


class AI_City(object):

    def __init__(self, root, data_type='reid', fps=10, trainval=False, gt_type='gt'):
        """
        Initialize the index.

        Args:
            self: (todo): write your description
            root: (str): write your description
            data_type: (str): write your description
            fps: (todo): write your description
            trainval: (todo): write your description
            gt_type: (str): write your description
        """
        if data_type == 'tracking_gt':
            self.root = osp.join(root, 'AIC19')
            if not trainval:
                train_dir = osp.join(root, f'AIC19/ALL_{gt_type}_bbox/train')
            else:
                train_dir = osp.join(root, f'AIC19/ALL_{gt_type}_bbox/trainval')
            val_dir = osp.join(root, f'AIC19/ALL_gt_bbox/val')
            self.train_path = osp.join(train_dir, f'gt_bbox_{fps}_fps')
            self.gallery_path = osp.join(val_dir, 'gt_bbox_1_fps')
            self.query_path = osp.join(val_dir, 'gt_bbox_1_fps')
        elif data_type == 'tracking_det':
            self.root = root
            self.train_path = root
            self.gallery_path = None
            self.query_path = None
        elif data_type == 'reid':  # reid
            self.root = osp.join(root, 'AIC19-reid')
            self.train_path = osp.join(root, 'AIC19-reid/image_train')
            self.gallery_path = osp.join(root, 'VeRi/image_query/')
            self.query_path = osp.join(root, 'VeRi/image_test/')

            xml_dir = osp.join(root, 'AIC19-reid/train_label.xml')
            self.reid_info = XD.parse(xml_dir).documentElement.getElementsByTagName('Item')
            self.index_by_fname_dict = defaultdict()
            for index in range(len(self.reid_info)):
                fname = self.reid_info[index].getAttribute('imageName')
                self.index_by_fname_dict[fname] = index
        elif data_type == 'reid_test':  # reid_test for feature extraction
            self.root = osp.join(root, 'AIC19-reid')
            self.train_path = None
            self.gallery_path = osp.join(root, 'AIC19-reid/image_test')
            self.query_path = osp.join(root, 'AIC19-reid/image_query')
        else:
            raise Exception

        self.train, self.query, self.gallery = [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids = 0, 0, 0
        self.num_cams = 40

        self.data_type = data_type
        self.load()

    def preprocess(self, path, relabel=True, type='reid'):
        """
        Preprocess a list of files.

        Args:
            self: (todo): write your description
            path: (str): write your description
            relabel: (todo): write your description
            type: (str): write your description
        """
        if type == 'tracking_det':
            pattern = re.compile(r'c([-\d]+)_f(\d+)')
        elif type == 'tracking_gt':
            pattern = re.compile(r'([-\d]+)_c(\d+)')
        else:  # reid
            pattern = None
        all_pids = {}
        ret = []
        if path is None:
            return ret, int(len(all_pids))
        fpaths = sorted(glob(osp.join(path, '*.jpg')))
        for fpath in fpaths:
            fname = osp.basename(fpath)
            if type == 'tracking_det':
                cam, frame = map(int, pattern.search(fname).groups())
                pid = 1
            elif type == 'tracking_gt':
                pid, cam = map(int, pattern.search(fname).groups())
            elif type == 'reid':  # reid
                pid, cam = map(int, [self.reid_info[self.index_by_fname_dict[fname]].getAttribute('vehicleID'),
                                     self.reid_info[self.index_by_fname_dict[fname]].getAttribute('cameraID')[1:]])
            else:  # reid test
                pid, cam = 1, 1
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
        Loads the gallery dataset.

        Args:
            self: (todo): write your description
        """
        self.train, self.num_train_ids = self.preprocess(self.train_path, True, self.data_type)
        self.gallery, self.num_gallery_ids = self.preprocess(self.gallery_path, False,
                                                             'reid_test' if self.data_type == 'reid_test' else 'tracking_gt')
        self.query, self.num_query_ids = self.preprocess(self.query_path, False,
                                                         'reid_test' if self.data_type == 'reid_test' else 'tracking_gt')

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:5d} | {:8d}"
              .format(self.num_train_ids, len(self.train)))
        print("  query    | {:5d} | {:8d}"
              .format(self.num_query_ids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}"
              .format(self.num_gallery_ids, len(self.gallery)))
