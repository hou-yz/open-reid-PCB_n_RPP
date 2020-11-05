from __future__ import absolute_import

from ..utils import to_torch


def accuracy(output, target, topk=(1,)):
    """
    Accuracy accuracy.

    Args:
        output: (todo): write your description
        target: (todo): write your description
        topk: (todo): write your description
    """
    output, target = to_torch(output), to_torch(target)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    ret = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
        ret.append(correct_k.mul_(1. / batch_size))
    return ret
