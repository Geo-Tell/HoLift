import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math


model_urls = {'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'}

class VGG(nn.Module):

    def __init__(self, features, num_classes=6):
        super(VGG, self).__init__()
        self.features = features
        self.num_classes = num_classes
        self.extra_convs = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True)          
        )
        self.extra_last_conv = nn.Conv2d(512,num_classes,1)
        self.proj_head = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(512, 256, kernel_size=1)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.extra_convs(x)
        x1 = self.proj_head(x)

        self.map1 = x1.clone()
        
        x=self.extra_last_conv(x)
        self.map2 = x.clone()
        
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = x.view(-1, self.num_classes)
        
        return self.map1,self.map2,x

    def get_heatmaps(self):
        return self.map1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_layers(cfg, input_channel=3, batch_norm=False):
    layers = []
    in_channels = input_channel
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        else:
            if i > 13:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, dilation=2, padding=2)            
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG_CAM(nn.Module):
    def __init__(self, pretrained=False, input_channel=3, **kwargs):
        super(VGG_CAM, self).__init__()
        self.vgg = VGG(make_layers(cfg['D1'], input_channel=input_channel), **kwargs)
        
        if pretrained:
            print("load vgg weights")
            print("#######################%%%%%%%%%%%%%%%%%%%%")
            self.vgg.load_state_dict(model_zoo.load_url(model_urls['vgg16']), strict=False)

    def forward(self, input):        
        feature, cam, score=self.vgg(input)
        feature = F.normalize(feature, p=2, dim=1)
        return score, cam, feature

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        for name, value in self.named_parameters():
            if 'extra' in name:
                if 'weight' in name:
                    groups[2].append(value)
                else:
                    groups[3].append(value)
            else:
                if 'weight' in name:
                    groups[0].append(value)
                else:
                    groups[1].append(value)
        return groups
