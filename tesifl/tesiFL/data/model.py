import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.stride = stride
        self.in_ch, self.out_ch = in_ch, out_ch

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        sc = x
        if self.stride != 1 or self.in_ch != self.out_ch:
            sc = x[:, :, ::self.stride, ::self.stride]
            pad = self.out_ch - self.in_ch
            sc = F.pad(sc, (0, 0, 0, 0, pad // 2, pad - pad // 2))
        return F.relu(out + sc)


class ResNet20(nn.Module):
    def __init__(self, num_classes=10, n=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._stage(16, 16, n, 1)
        self.layer2 = self._stage(16, 32, n, 2)
        self.layer3 = self._stage(32, 64, n, 2)
        self.fc = nn.Linear(64, num_classes)

    @staticmethod
    def _stage(in_ch, out_ch, blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        layers += [BasicBlock(out_ch, out_ch) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)

class BlockR18(nn.Module):
    """Blocco base di ResNet-18 (He et al. 2016): due conv 3x3 con BN, e
    scorciatoia con conv 1x1 + BN quando cambia la dimensione."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    """ResNet-18 nella variante per CIFAR (32x32): primo strato conv 3x3 a
    stride 1 e niente max-pooling iniziale, quattro stadi da due blocchi
    (64, 128, 256, 512 canali). 11.22 M parametri e 555 MMAC di forward per
    immagine con 100 classi (experiment.toml, [model.specs])."""

    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._stage(64, 64, 1)
        self.layer2 = self._stage(64, 128, 2)
        self.layer3 = self._stage(128, 256, 2)
        self.layer4 = self._stage(256, 512, 2)
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _stage(in_ch, out_ch, stride):
        return nn.Sequential(BlockR18(in_ch, out_ch, stride), BlockR18(out_ch, out_ch))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


MODELS = {"resnet20": ResNet20, "resnet18": ResNet18}


def build_model(name: str = "resnet20", num_classes: int = 10):
    if name not in MODELS:
        raise ValueError(f"modello sconosciuto: {name} (usa {', '.join(MODELS)})")
    return MODELS[name](num_classes=num_classes)