"""This module contains different model implementations, in order to run experiments of the meta-evaluation framework."""

# This file is part of MetaQuantus.
# MetaQuantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# MetaQuantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with MetaQuantus. If not, see <https://www.gnu.org/licenses/>.

import torch


class LeNet(torch.nn.Module):
    """LeNet model with torch.nn"""

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True
        )
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0,
            bias=True,
        )
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.max_pool_1(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.max_pool_2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ResidualBlock(torch.nn.Module):
    """
    A residual block as defined by He et al.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ResidualBlock, self).__init__()
        self.conv_res1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
        )
        self.conv_res1_bn = torch.nn.BatchNorm2d(
            num_features=out_channels, momentum=0.9
        )
        self.conv_res2 = torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.conv_res2_bn = torch.nn.BatchNorm2d(
            num_features=out_channels, momentum=0.9
        )

        if stride != 1:
            # in case stride is not set to 1, we need to downsample the residual so that
            # the dimensions are the same when we add them together
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                torch.nn.BatchNorm2d(num_features=out_channels, momentum=0.9),
            )
        else:
            self.downsample = None

        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv_res1_bn(self.conv_res1(x)))
        out = self.conv_res2_bn(self.conv_res2(out))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.relu(out)
        out += residual
        return out


class ResNet9(torch.nn.Module):
    """
    A Residual network.
    """

    def __init__(self, nr_channels=3, nr_classes=10):
        super(ResNet9, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=nr_channels,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(num_features=64, momentum=0.9),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(num_features=128, momentum=0.9),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(num_features=256, momentum=0.9),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(num_features=256, momentum=0.9),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = torch.nn.Linear(in_features=1024, out_features=nr_classes, bias=True)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
        out = self.fc(out)
        return out
