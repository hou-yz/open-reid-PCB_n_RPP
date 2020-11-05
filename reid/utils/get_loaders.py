import os.path as osp
from torch import nn
from torch.utils.data import DataLoader
from reid import datasets
from reid.utils.serialization import load_checkpoint
from reid.utils.data.og_sampler import RandomIdentitySampler
from reid.utils.data.zju_sampler import ZJU_RandomIdentitySampler
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor



def get_data(name, data_dir, height, width, batch_size, workers,
             combine_trainval, crop, tracking_icams, fps, re=0, num_instances=0, camstyle=0, zju=0, colorjitter=0):
    """
    Create training data.

    Args:
        name: (str): write your description
        data_dir: (str): write your description
        height: (bool): write your description
        width: (int): write your description
        batch_size: (int): write your description
        workers: (str): write your description
        combine_trainval: (bool): write your description
        crop: (list): write your description
        tracking_icams: (bool): write your description
        fps: (bool): write your description
        re: (str): write your description
        num_instances: (int): write your description
        camstyle: (bool): write your description
        zju: (bool): write your description
        colorjitter: (array): write your description
    """
    # if name == 'market1501':
    #     root = osp.join(data_dir, 'Market-1501-v15.09.15')
    # elif name == 'duke_reid':
    #     root = osp.join(data_dir, 'DukeMTMC-reID')
    # elif name == 'duke_tracking':
    #     root = osp.join(data_dir, 'DukeMTMC')
    # else:
    #     root = osp.join(data_dir, name)
    if name == 'duke_tracking':
        if tracking_icams != 0:
            tracking_icams = [tracking_icams]
        else:
            tracking_icams = list(range(1, 9))
        dataset = datasets.create(name, data_dir, data_type='tracking_gt', iCams=tracking_icams, fps=fps,
                                  trainval=combine_trainval)
    elif name == 'aic_tracking':
        dataset = datasets.create(name, data_dir, data_type='tracking_gt', fps=fps, trainval=combine_trainval)
    else:
        dataset = datasets.create(name, data_dir)
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    num_classes = dataset.num_train_ids

    train_transformer = T.Compose([
        T.ColorJitter(brightness=0.1 * colorjitter, contrast=0.1 * colorjitter, saturation=0.1 * colorjitter, hue=0),
        T.Resize((height, width)),
        T.RandomHorizontalFlip(),
        T.Pad(10 * crop),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=re),
    ])
    test_transformer = T.Compose([
        T.Resize((height, width)),
        # T.RectScale(height, width, interpolation=3),
        T.ToTensor(),
        normalizer,
    ])

    if zju:
        train_loader = DataLoader(
            Preprocessor(dataset.train, root=dataset.train_path, transform=train_transformer),
            batch_size=batch_size, num_workers=workers,
            sampler=ZJU_RandomIdentitySampler(dataset.train, batch_size, num_instances) if num_instances else None,
            shuffle=False if num_instances else True, pin_memory=True, drop_last=False if num_instances else True)
    else:
        train_loader = DataLoader(
            Preprocessor(dataset.train, root=dataset.train_path, transform=train_transformer),
            batch_size=batch_size, num_workers=workers,
            sampler=RandomIdentitySampler(dataset.train, num_instances) if num_instances else None,
            shuffle=False if num_instances else True, pin_memory=True, drop_last=True)
    query_loader = DataLoader(
        Preprocessor(dataset.query, root=dataset.query_path, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery, root=dataset.gallery_path, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    if camstyle <= 0:
        camstyle_loader = None
    else:
        camstyle_loader = DataLoader(
            Preprocessor(dataset.camstyle, root=dataset.camstyle_path, transform=train_transformer),
            batch_size=camstyle, num_workers=workers,
            shuffle=True, pin_memory=True, drop_last=True)
    return dataset, num_classes, train_loader, query_loader, gallery_loader, camstyle_loader


def checkpoint_loader(model, path):
    """
    Checkpoint checkpoint.

    Args:
        model: (todo): write your description
        path: (str): write your description
    """
    checkpoint = load_checkpoint(path)
    pretrained_dict = checkpoint['state_dict']
    if isinstance(model, nn.DataParallel):
        Parallel = 1
        model = model.module.cpu()
    else:
        Parallel = 0

    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # if eval_only:
    #     keys_to_del = []
    #     for key in pretrained_dict.keys():
    #         if 'classifier' in key:
    #             keys_to_del.append(key)
    #     for key in keys_to_del:
    #         del pretrained_dict[key]
    #     pass
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    start_epoch = checkpoint['epoch']
    best_top1 = checkpoint['best_top1']

    if Parallel:
        model = nn.DataParallel(model).cuda()

    return model, start_epoch, best_top1
