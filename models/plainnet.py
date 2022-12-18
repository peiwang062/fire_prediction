import torch.nn as nn
import math

__all__ = ['PlainNet', 'plainnet20', 'plainnet56', 'plainnet8']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.stride = stride
#         # self.bn_unaff = nn.BatchNorm2d(planes, affine=False)
#
#
#     def forward(self, x):
#         out1 = self.conv1(x)
#         bn1 = self.bn1(out1)
#         ac1 = self.relu(bn1)
#
#         out2 = self.conv2(ac1)
#         bn2 = self.bn2(out2)
#         ac2 = self.relu(bn2)
#
#         return ac2, out2, self.bn2.running_mean, self.bn2.running_var
#
#
# class PlainNet(nn.Module):
#
#     def __init__(self, block, layers, inplanes=16, num_classes=10):
#         self.inplanes = inplanes
#         super(PlainNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, inplanes, kernel_size=3, stride=1, padding=1,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.layer1 = self._make_layer(block, inplanes, layers[0])
#         self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
#         self.avgpool = nn.AvgPool2d(8, stride=1)
#         self.fc = nn.Linear(inplanes * 4, num_classes)
#
#
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 if m.affine == True:
#                     m.weight.data.fill_(1)
#                     m.bias.data.zero_()
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         layers = []
#         layers.append(block(self.inplanes, planes, stride))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         ac1 = self.conv1(x)
#         bn1 = self.bn1(ac1)
#         x = self.relu(bn1)
#
#
#         bn1_m = self.bn1.running_mean.view(1, -1, 1, 1)
#         bn1_v = self.bn1.running_var.sqrt().view(1, -1, 1, 1)
#
#         # print (ac1, bn1_m)
#         bn1 = (ac1.data - bn1_m) / bn1_v
#
#         x, ac3, bn3_m, bn3_v = self.layer1(x)
#         x, ac5, bn5_m, bn5_v = self.layer2(x)
#         x, ac7, bn7_m, bn7_v = self.layer3(x)
#
#         bn3 = (ac3.data - bn3_m.view(1, -1, 1, 1)) / bn3_v.sqrt().view(1, -1, 1, 1)
#         bn5 = (ac5.data - bn5_m.view(1, -1, 1, 1)) / bn5_v.sqrt().view(1, -1, 1, 1)
#         bn7 = (ac7.data - bn7_m.view(1, -1, 1, 1)) / bn7_v.sqrt().view(1, -1, 1, 1)
#
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#
#         return x, [ac1, ac3, ac5, ac7], [bn1, bn3, bn5, bn7]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class PlainNet(nn.Module):

    def __init__(self, block, layers, inplanes=16, num_classes=10):
        self.inplanes = inplanes
        super(PlainNet, self).__init__()
        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(inplanes * 4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x





def plainnet20(inplanes, num_classes):
    model = PlainNet(BasicBlock, [3, 3, 3], inplanes, num_classes)
    return model


def plainnet56(inplanes, num_classes):
    model = PlainNet(BasicBlock, [9, 9, 9], inplanes, num_classes)
    return model

def plainnet8(inplanes, num_classes):
    model = PlainNet(BasicBlock, [1, 1, 1], inplanes, num_classes)
    return model