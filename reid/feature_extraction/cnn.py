from __future__ import absolute_import

from collections import OrderedDict

import torch
from torch.autograd import Variable

from ..utils import to_torch


def extract_cnn_feature(model, inputs, modules=None):
    """
    Extract a feature.

    Args:
        model: (todo): write your description
        inputs: (array): write your description
        modules: (list): write your description
    """
    model.eval()
    inputs = to_torch(inputs)
    inputs = Variable(inputs, requires_grad=False)
    if modules is None:
        # if isinstance(model.module, IDE_model) or isinstance(model.module, PCB_model):
        with torch.no_grad():
            outputs = model(inputs)
        outputs = outputs[0]
        # else:
        #     outputs = model(inputs)
        outputs = outputs.data.cpu()
        return outputs
    # Register forward hook for each module
    outputs = OrderedDict()
    handles = []
    for m in modules:
        outputs[id(m)] = None

        """
        Return the function

        Args:
            m: (int): write your description
            i: (int): write your description
            o: (int): write your description
        """
        def func(m, i, o): outputs[id(m)] = o.data.cpu()

        handles.append(m.register_forward_hook(func))
    model(inputs)
    for h in handles:
        h.remove()
    return list(outputs.values())
