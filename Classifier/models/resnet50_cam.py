import torch
import torch.nn as nn
import torch.nn.functional as F
from Classifier.models import resnet50

class Net(nn.Module):
    def __init__(self, num_cls):
        super(Net, self).__init__()
        self.num_cls = num_cls
        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.classifier = nn.Conv2d(2048, self.num_cls, 1, bias=False)

        self.proj_head = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(512, 256, kernel_size=1)
        )

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier, self.proj_head])

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x1 = self.proj_head(x)
        feature = x1.clone()
        feat_proj = F.normalize(feature, p=2, dim=1)

        x = self.classifier(x)
        cam = x.clone()
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = x.view(-1, self.num_cls)
        return x, cam, feat_proj
    
    def trainable_parameters(self):
        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))
        