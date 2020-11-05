import numpy as np
import torch
import torch.nn.functional as F
from reid.evaluators import evaluate_all, pairwise_distance


def metric_distance(model, query_features, gallery_features):
    """
    Calculate distance between features.

    Args:
        model: (todo): write your description
        query_features: (str): write your description
        gallery_features: (bool): write your description
    """
    dist = np.zeros([len(query_features), len(gallery_features)])
    step = 1024
    for i in range(len(query_features)):
        for j in range(0, len(gallery_features), step):
            numel = min(step, len(gallery_features) - j)
            query_feat = query_features[i].view([1, -1]).repeat([numel, 1]).cuda()
            gallery_feat = gallery_features[j:j + numel].cuda()
            output = model(query_feat, gallery_feat)
            dist[i, j:j + numel] = F.softmax(output, dim=1)[:, 0].cpu().detach().numpy()
    return dist


def metric_evaluate(model, query_set, gallery_set):
    """
    Evaluate the model.

    Args:
        model: (str): write your description
        query_set: (todo): write your description
        gallery_set: (todo): write your description
    """
    model.eval()
    print('=> L2 distance')
    dist = pairwise_distance(query_set.features, gallery_set.features)
    evaluate_all(dist, query_ids=query_set.labels[:, 1], gallery_ids=gallery_set.labels[:, 1],
                 query_cams=query_set.labels[:, 0], gallery_cams=gallery_set.labels[:, 0], )
    print('=> Metric')
    dist = metric_distance(model, query_set.features, gallery_set.features)
    evaluate_all(dist, query_ids=query_set.labels[:, 1], gallery_ids=gallery_set.labels[:, 1],
                 query_cams=query_set.labels[:, 0], gallery_cams=gallery_set.labels[:, 0], )
    return
