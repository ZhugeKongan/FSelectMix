'''
paper:https://arxiv.org/abs/1611.05431
github:https://github.com/miraclewkf/ResNeXt-PyTorch
New for ResNeXt:
1. Wider bottleneck
2. Add group for conv2
'''

import torch.nn as nn
import math

__all__ = ['ResNeXt', 'resnext18', 'resnext34', 'resnext50', 'resnext101',
           'resnext152']

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def Conv1(in_planes, places, k_size=3,stride=1,padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=k_size,stride=stride,padding=padding, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
class ESE(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ESE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.fc = nn.Linear(channel, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        # y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.fc(y.view(x.size(0), -1)).view(x.size(0), -1, 1, 1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return y#x * y.expand_as(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes*2, stride)
        self.bn1 = nn.BatchNorm2d(planes*2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes*2, planes*2, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes*2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes*2)
        self.conv2 = nn.Conv2d(planes*2, planes*2, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.conv3 = nn.Conv2d(planes*2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self, block, layers, num_classes=1000, num_group=32):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = Conv1(3, 64)

        self.se0 = ESE(64)
        self.se1 = ESE(64 * block.expansion)
        self.se2 = ESE(128 * block.expansion)
        self.se3 = ESE(256 * block.expansion)

        self.layer1 = self._make_layer(block, 64, layers[0], num_group)
        self.layer2 = self._make_layer(block, 128, layers[1], num_group, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], num_group, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], num_group, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.classifer = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, num_group, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, num_group=num_group))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_group=num_group))

        return nn.Sequential(*layers)

    def forward(self, x,rand_index=None,r=1):
        b = x.size(0)
        x = self.conv1(x)

        se0 = self.se0(x)
        x = x * se0
        if  r < 0.0 and r>=0.0:
                r1 = torch.argsort(se0, dim=1, descending=True).view(b, -1).cuda()
                rand2 = r1[:, ::2]
                rand2, _ = torch.sort(rand2, dim=1)
                f=x[rand_index].clone()
                for i in range(b):
                    x[i][rand2[i]] = f[i][rand2[i]]

        x = self.layer1(x)
        se1 = self.se1(x)
        x = x * se1
        if  r < 0.3 and r>=0.2:
                r1 = torch.argsort(se1, dim=1, descending=True).view(b, -1).cuda()
                rand2 = r1[:, ::2]
                rand2, _ = torch.sort(rand2, dim=1)
                f=x[rand_index].clone()
                for i in range(b):
                    x[i][rand2[i]] = f[i][rand2[i]]

        x = self.layer2(x)
        se2 = self.se2(x)
        x = x * se2
        if  r < 0.4 and r>=0.3:
                r1 = torch.argsort(se2, dim=1, descending=True).view(b, -1).cuda()
                rand2 = r1[:, ::2]
                rand2, _ = torch.sort(rand2, dim=1)
                f=x[rand_index].clone()
                for i in range(b):
                    x[i][rand2[i]] = f[i][rand2[i]]

        x = self.layer3(x)
        se3 = self.se3(x)
        x = x * se3
        if  r < 0.5 and r>=0.4:
                r1 = torch.argsort(se3, dim=1, descending=True).view(b, -1).cuda()
                rand2 = r1[:, ::2]
                rand2, _ = torch.sort(rand2, dim=1)
                f=x[rand_index].clone()
                for i in range(b):
                    x[i][rand2[i]] = f[i][rand2[i]]
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x1 = self.fc(x)
        x2 = self.classifer(x)

        return x1,x2


def resnext18( **kwargs):
    model = ResNeXt(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnext34(**kwargs):
    model = ResNeXt(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnext50(**kwargs):
    model = ResNeXt(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnext101(**kwargs):
    model = ResNeXt(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnext152(**kwargs):
    model = ResNeXt(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

import torch
if __name__ == '__main__':
    net = resnext50(num_classes=200)
    y,_ = net(torch.randn(10, 3, 32, 32))
    # print(net)
    print(y.size())
    from thop import profile
    input = torch.randn(1, 3, 64, 64)
    flops, params = profile(net, inputs=(input, ))
    total = sum([param.nelement() for param in net.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
    print('  + Number of params: %.3fG' % (flops / 1e9))
    print('flops: ', flops, 'params: ', params)