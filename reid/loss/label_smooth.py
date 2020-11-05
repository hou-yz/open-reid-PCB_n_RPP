from __future__ import print_function, absolute_import

import torch
from torch import nn


class LSR_loss(nn.Module):

    def __init__(self, e=0.1):
        """
        Initialize logSoft.

        Args:
            self: (todo): write your description
            e: (int): write your description
        """
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e

    def _one_hot(self, labels, classes, value=1):
        """
        One - hot - hotch labels.

        Args:
            self: (todo): write your description
            labels: (todo): write your description
            classes: (list): write your description
            value: (str): write your description
        """
        one_hot = torch.zeros(labels.size(0), classes)
        # labels and value_added  size must match
        labels = labels.view(labels.size(0), -1)
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)
        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)
        one_hot.scatter_add_(1, labels, value_added)
        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        """
        Construct a target label.

        Args:
            self: (todo): write your description
            target: (todo): write your description
            length: (int): write your description
            smooth_factor: (int): write your description
        """
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        one_hot += smooth_factor / length
        return one_hot.to(target.device)

    def forward(self, x, target):
        """
        Forward loss.

        Args:
            self: (todo): write your description
            x: (todo): write your description
            target: (todo): write your description
        """
        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        x = self.log_softmax(x)
        loss = torch.sum(- x * smoothed_target, dim=1)
        return torch.mean(loss)
