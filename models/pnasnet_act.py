'''PNASNet in PyTorch.

Paper: Progressive Neural Architecture Search
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class SepConv_act(nn.Module):
    '''Separable Convolution.'''
    def __init__(self, in_planes, out_planes, kernel_size, stride, activation=F.relu):
        super(SepConv_act, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes,
                               kernel_size, stride,
                               padding=(kernel_size-1)//2,
                               bias=False, groups=in_planes)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.activation = activation

    def forward(self, x):
        return self.bn1(self.conv1(x))


class CellA_act(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, activation=F.relu):
        super(CellA_act, self).__init__()
        self.stride = stride
        self.sep_conv1 = SepConv_act(in_planes, out_planes, kernel_size=7, stride=stride)
        if stride==2:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(out_planes)
        self.activation = activation

    def forward(self, x):
        y1 = self.sep_conv1(x)
        y2 = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)
        if self.stride==2:
            y2 = self.bn1(self.conv1(y2))
        return self.activation(y1+y2)

class CellB_act(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, activation=F.relu):
        super(CellB_act, self).__init__()
        self.stride = stride
        # Left branch
        self.sep_conv1 = SepConv_act(in_planes, out_planes, kernel_size=7, stride=stride)
        self.sep_conv2 = SepConv_act(in_planes, out_planes, kernel_size=3, stride=stride)
        # Right branch
        self.sep_conv3 = SepConv_act(in_planes, out_planes, kernel_size=5, stride=stride)
        if stride==2:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(out_planes)
        # Reduce channels
        self.conv2 = nn.Conv2d(2*out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.activation = activation

    def forward(self, x):
        # Left branch
        y1 = self.sep_conv1(x)
        y2 = self.sep_conv2(x)
        # Right branch
        y3 = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)
        if self.stride==2:
            y3 = self.bn1(self.conv1(y3))
        y4 = self.sep_conv3(x)
        # Concat & reduce channels
        b1 = self.activation(y1+y2)
        b2 = self.activation(y3+y4)
        y = torch.cat([b1,b2], 1)
        return self.activation(self.bn2(self.conv2(y)))

class PNASNet_act(nn.Module):
    def __init__(self, cell_type, num_cells, num_planes, activation=F.relu):
        super(PNASNet_act, self).__init__()
        self.in_planes = num_planes
        self.cell_type = cell_type
        self.activation=activation

        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_planes)

        self.layer1 = self._make_layer(num_planes, num_cells=6)
        self.layer2 = self._downsample(num_planes*2)
        self.layer3 = self._make_layer(num_planes*2, num_cells=6)
        self.layer4 = self._downsample(num_planes*4)
        self.layer5 = self._make_layer(num_planes*4, num_cells=6)

        self.linear = nn.Linear(num_planes*4, 10)

    def _make_layer(self, planes, num_cells):
        layers = []
        for _ in range(num_cells):
            layers.append(self.cell_type(self.in_planes, planes, stride=1))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _downsample(self, planes):
        layer = self.cell_type(self.in_planes, planes, stride=2)
        self.in_planes = planes
        return layer

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = F.avg_pool2d(out, 8)
        out = self.linear(out.view(out.size(0), -1))
        return out


def PNASNetA_act(activation=F.relu):
    return PNASNet_act(CellA_act, num_cells=6, num_planes=44, activation=activation)

def PNASNetB_act(activation=F.relu):
    return PNASNet_act(CellB_act, num_cells=6, num_planes=32, activation=activation)


def test():
    net = PNASNetB_act(activation=F.relu)
    print(net)
    x = Variable(torch.randn(1,3,32,32))
    y = net(x)
    print(y)

# test()
