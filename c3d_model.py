import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import torch.utils.model_zoo as model_zoo
from transforms import *

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1),
                             out.size(2), out.size(3),
                             out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, modality='RGB',shortcut_type='B', num_classes=101,temporal_segs=4):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2),
        #                        padding=(3, 3, 3), bias=False)
        self.input_channel = 3 if modality=='RGB' else 2
        self.conv1_custom = nn.Conv3d(self.input_channel, 64, kernel_size=7, stride=(1,2,2),
                                padding=(3,3,3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        self.avgpool = nn.AvgPool3d((1, 4, 4), stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # some private attribute
        self.input_size=112 # input of the network
        self.in_temporal=10 # temporal length
        self.temporal_segs=temporal_segs # num of temporals to ST-GAP
        self.input_mean = [0.485, 0.456, 0.406] if modality=='RGB' else [0.5]
        self.input_std = [0.229, 0.224, 0.225] if modality=='RGB' else [np.mean([0.229, 0.224, 0.225])]


    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    @property
    def crop_size(self):
        return self.input_size

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_custom(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        x = self.avgpool(x)

        x = x.view(-1,self.temporal_segs,self.fc.in_features).mean(1)

        x = self.fc(x)

        return x

def filter2d_to_3d(weight2d, weight3d):
    nb_filter, channel, h, w, c = weight3d.size()
    for i in range(c):
        weight3d[:,:,:,:,i] = weight2d.data/float(c)
    
    return weight3d

        
def cross_modality_pretrain(conv1_weight, channel=3):
    # transform the original 3 channel weight to "channel" channel
    S=0
    for i in range(3):
        S += conv1_weight[:,i,:,:]
    avg = S/3.
    new_conv1_weight = torch.FloatTensor(64,channel,7,7)
    #print type(avg),type(new_conv1_weight)
    for i in range(channel):
        new_conv1_weight[:,i,:,:] = avg.data  # every channel set a avg value
    return new_conv1_weight

def weight_trainsform(pretrain_dict, model_dict, input_channel=3):

    if input_channel!=3:
        w3 = pretrain_dict['conv1.weight']
        wt = cross_modality_pretrain(w3,channel=input_channel)
        model_dict['custom_conv1.weight']=wt

    for i,key in enumerate(pretrain_dict):
        if i==0:continue
        if key.split('.')[-2][:-1] == 'conv':
            #print key,pretrain_dict[key].size()
            weight2d = pretrain_dict[key]
            weight3d = model_dict[key]
            model_dict[key] = filter2d_to_3d(weight2d, weight3d)

    return model_dict
 

def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    pretrain_dict = model_zoo.load_url(model_urls['resnet18'])
    model_dict = model.state_dict()
    weight_dict = weight_trainsform(pretrain_dict,model_dict,model.input_channel)
    model.load_state_dict(weight_dict)

    return model

def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    pretrain_dict = model_zoo.load_url(model_urls['resnet34'])
    model_dict = model.state_dict()
    weight_dict = weight_trainsform(pretrain_dict,model_dict,model.input_channel)
    model.load_state_dict(weight_dict)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    pretrain_dict = model_zoo.load_url(model_urls['resnet50'])
    model_dict = model.state_dict()
    weight_dict = weight_trainsform(pretrain_dict,model_dict,model.input_channel)
    model.load_state_dict(weight_dict)
    return model

def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    pretrain_dict = model_zoo.load_url(model_urls['resnet101'])
    model_dict = model.state_dict()
    weight_dict = weight_trainsform(pretrain_dict,model_dict,model.input_channel)
    model.load_state_dict(weight_dict)
    return model


# custom operation
def get_optim_policies(model=None,modality='RGB',enable_pbn=True):
    '''
    first conv:         weight --> conv weight
                        bias   --> conv bias
    normal action:      weight --> non-first conv + fc weight
                        bias   --> non-first conv + fc bias
    bn:                 the first bn2, and many all bn1.

    '''
    first_conv_weight = []
    first_conv_bias = []
    normal_weight = []
    normal_bias = []
    bn = []

    if model==None:
        print 'no model!'
        exit()

    conv_cnt = 0
    bn_cnt = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Conv2d):
            ps = list(m.parameters())
            conv_cnt += 1
            if conv_cnt == 1:
                first_conv_weight.append(ps[0])
                if len(ps) == 2:
                    first_conv_bias.append(ps[1])
            else:
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
        elif isinstance(m, torch.nn.Linear):
            ps = list(m.parameters())
            normal_weight.append(ps[0])
            if len(ps) == 2:
                normal_bias.append(ps[1])
              
        elif isinstance(m, torch.nn.BatchNorm1d):
            bn.extend(list(m.parameters()))
        elif isinstance(m, torch.nn.BatchNorm3d):
            bn_cnt += 1
            # later BN's are frozen
            if not enable_pbn or bn_cnt == 1:
                bn.extend(list(m.parameters()))
        elif len(m._modules) == 0:
            if len(list(m.parameters())) > 0:
                raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

    return [
        {'params': first_conv_weight, 'lr_mult': 5 if modality == 'Flow' else 1, 'decay_mult': 1,
         'name': "first_conv_weight"},
        {'params': first_conv_bias, 'lr_mult': 10 if modality == 'Flow' else 2, 'decay_mult': 0,
         'name': "first_conv_bias"},
        {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
         'name': "normal_weight"},
        {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
         'name': "normal_bias"},
        {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
         'name': "BN scale/shift"},
    ]


def get_augmentation(model,modality):
    if modality=='RGB':
        return torchvision.transforms.Compose([
            GroupMultiScaleCrop(model.input_size, [1, .875, .75, .66]),
            GroupRandomHorizontalFlip(is_flow=False)
            ])
    else: # modality=='Flow':
        return torchvision.transforms.Compose([
            GroupMultiScaleCrop(model.input_size, [1, .875, .75]),
            GroupRandomHorizontalFlip(is_flow=True)])


if __name__ == '__main__':
    model = resnet18()
    model = resnet34()
    model = resnet50()
    model = resnet101()