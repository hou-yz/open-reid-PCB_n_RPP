from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torchvision.models import resnet50, densenet121


class IDE_model(nn.Module):
    def __init__(self, feature_dim=256, num_classes=0, norm=False, dropout=0, last_stride=2, arch='resnet50'):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            feature_dim: (str): write your description
            num_classes: (int): write your description
            norm: (todo): write your description
            dropout: (str): write your description
            last_stride: (str): write your description
            arch: (todo): write your description
        """
        super(IDE_model, self).__init__()
        # Create IDE_only model
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.norm = norm

        if arch == 'resnet50':
            self.base = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
            if last_stride != 2:
                # decrease the downsampling rate
                # change the stride2 conv layer in self.layer4 to stride=1
                self.base[7][0].conv2.stride = last_stride
                # change the downsampling layer in self.layer4 to stride=1
                self.base[7][0].downsample[0].stride = last_stride
            base_dim = 2048
        elif arch == 'densenet121':
            self.base = nn.Sequential(*list(densenet121(pretrained=True).children())[:-1])[0]
            if last_stride != 2:
                # remove the pooling layer in last transition block
                self.base[-3][-1].stride = 1
                self.base[-3][-1].kernel_size = 1
                pass
            base_dim = 1024
        else:
            raise Exception('Please select arch from [resnet50, densenet121]!')

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # feat & feat_bn
        if self.feature_dim > 0:
            self.feat_fc = nn.Linear(base_dim, feature_dim)
            init.kaiming_normal_(self.feat_fc.weight, mode='fan_out')
            init.constant_(self.feat_fc.bias, 0.0)
        else:
            feature_dim = base_dim

        self.feat_bn = nn.BatchNorm1d(feature_dim)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        # dropout before classifier
        self.dropout = dropout
        if self.dropout > 0:
            self.drop_layer = nn.Dropout2d(self.dropout)

        # classifier:
        if self.num_classes > 0:
            self.classifier = nn.Linear(feature_dim, self.num_classes, bias=False)
            init.normal_(self.classifier.weight, std=0.001)
        pass

    def forward(self, x):
        """
        Returns:
          h_s: each member with shape [N, c]
          prediction_s: each member with shape [N, num_classes]
        """
        x = self.base(x)
        x = self.global_avg_pool(x).view(x.shape[0], -1)
        feature_base = x

        if self.feature_dim > 0:
            x = self.feat_fc(x)
            feature_ide = x
        else:
            feature_ide = feature_base

        x = self.feat_bn(x)
        # no relu after feature_fc

        if self.dropout > 0:
            x = self.drop_layer(x)

        prediction_s = []
        if self.num_classes > 0 and self.training:
            prediction = self.classifier(x)
            prediction_s.append(prediction)

        if self.norm:
            feature_ide = F.normalize(feature_ide)

        return feature_ide, tuple(prediction_s)
