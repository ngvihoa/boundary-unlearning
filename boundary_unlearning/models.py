"""AllCNN architecture used by the author's CIFAR-10 experiment."""

from torch import nn


class Identity(nn.Module):
    def forward(self, x):
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, activation_fn=nn.ReLU, batch_norm=True):
        if padding is None:
            padding = (kernel_size - 1) // 2
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not batch_norm)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels, affine=True))
        layers.append(activation_fn())
        super().__init__(*layers)


class AllCNN(nn.Module):
    def __init__(self, n_channels=3, num_classes=10, filters_percentage=0.5, batch_norm=True):
        super().__init__()
        n_filters_1 = int(96 * filters_percentage)
        n_filters_2 = int(192 * filters_percentage)
        self.features = nn.Sequential(
            Conv(n_channels, n_filters_1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filters_1, n_filters_1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filters_1, n_filters_2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
            Identity(),
            Conv(n_filters_2, n_filters_2, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filters_2, n_filters_2, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filters_2, n_filters_2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
            Identity(),
            Conv(n_filters_2, n_filters_2, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filters_2, n_filters_2, kernel_size=1, batch_norm=batch_norm),
            nn.AvgPool2d(8),
            Flatten(),
        )
        self.classifier = nn.Sequential(nn.Linear(n_filters_2, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def init_linear_like_repo(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
