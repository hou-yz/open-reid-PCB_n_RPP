from __future__ import absolute_import

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=1):
        """
        Initialize the instances.

        Args:
            self: (todo): write your description
            data_source: (str): write your description
            num_instances: (int): write your description
        """
        super(RandomIdentitySampler, self).__init__(data_source)
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Args:
            self: (todo): write your description
        """
        return self.num_samples * self.num_instances

    def __iter__(self):
        """
        Iterate over the instance.

        Args:
            self: (todo): write your description
        """
        indices = torch.randperm(self.num_samples)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)
