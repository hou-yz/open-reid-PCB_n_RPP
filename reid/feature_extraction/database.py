from __future__ import absolute_import

import h5py
import numpy as np
from torch.utils.data import Dataset


class FeatureDatabase(Dataset):
    def __init__(self, *args, **kwargs):
        """
        Initialize an h5 file.

        Args:
            self: (todo): write your description
        """
        super(FeatureDatabase, self).__init__()
        self.fid = h5py.File(*args, **kwargs)

    def __enter__(self):
        """
        Decor function.

        Args:
            self: (todo): write your description
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Called when an exception is raised.

        Args:
            self: (todo): write your description
            exc_type: (todo): write your description
            exc_val: (todo): write your description
            exc_tb: (todo): write your description
        """
        self.close()

    def __getitem__(self, keys):
        """
        Get a single item from the cache.

        Args:
            self: (todo): write your description
            keys: (str): write your description
        """
        if isinstance(keys, (tuple, list)):
            return [self._get_single_item(k) for k in keys]
        return self._get_single_item(keys)

    def _get_single_item(self, key):
        """
        Return the single item for a single item.

        Args:
            self: (todo): write your description
            key: (str): write your description
        """
        return np.asarray(self.fid[key])

    def __setitem__(self, key, value):
        """
        Sets a dataset

        Args:
            self: (todo): write your description
            key: (str): write your description
            value: (str): write your description
        """
        if key in self.fid:
            if self.fid[key].shape == value.shape and \
                    self.fid[key].dtype == value.dtype:
                self.fid[key][...] = value
            else:
                del self.fid[key]
                self.fid.create_dataset(key, data=value)
        else:
            self.fid.create_dataset(key, data=value)

    def __delitem__(self, key):
        """
        Remove an item from cache

        Args:
            self: (todo): write your description
            key: (str): write your description
        """
        del self.fid[key]

    def __len__(self):
        """
        Returns the length of the queue.

        Args:
            self: (todo): write your description
        """
        return len(self.fid)

    def __iter__(self):
        """
        Returns an iterator over the iterator

        Args:
            self: (todo): write your description
        """
        return iter(self.fid)

    def flush(self):
        """
        Flush the cache.

        Args:
            self: (todo): write your description
        """
        self.fid.flush()

    def close(self):
        """
        Close the connection.

        Args:
            self: (todo): write your description
        """
        self.fid.close()
