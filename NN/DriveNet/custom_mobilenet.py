import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


class CustomMobileNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(CustomMobileNet, self).__init__()

        base_net = mobilenet_v2(pretrained=pretrained)

        self.features = base_net.features
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 256),
        )
        self.concat_fc = nn.Linear(260, num_classes)
        # self.concat_fc = nn.Sequential(
        #     nn.Linear(260, 64),
        #     nn.Dropout(0.3),
        #     nn.Linear(64, num_classes)
        # )

        # nn.init.normal_(self.concat_fc.weight, 0, 0.01)
        # nn.init.zeros_(self.concat_fc.bias)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x, actions):

        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        x = torch.cat([x, actions], dim=1)
        x = self.concat_fc(x)
        return x
        
        
class CustomMobileNetExt(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(CustomMobileNetExt, self).__init__()

        base_net = mobilenet_v2(pretrained=pretrained)

        self.features = base_net.features
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 251),
        )
        self.concat_fc = nn.Linear(256, num_classes)

        # nn.init.normal_(self.concat_fc.weight, 0, 0.01)
        # nn.init.zeros_(self.concat_fc.bias)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x, actions, prev_throttle):

        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        x = torch.cat([x, actions, prev_throttle], dim=1)
        x = self.concat_fc(x)
        return x
