import torch
import torch.nn as nn
import torch.nn.functional as F


class MSIA(nn.Module):
    """
    Multi-scale Illumination Attention Module
    This module performs multi-scale illumination attention to capture the illumination information of the input image.
    """

    def __init__(self, filters, activation='lrelu'):
        super().__init__()
        # Down1
        self.conv_bn_relu1 = ConvBNReLU(filters, activation)

        # Down2
        self.down_2 = MaxPooling2D(2,2)
        self.conv_bn_relu2 = ConvBNReLU(filters,activation)
        self.deconv_2 = ConvTranspose2D(filters, filters)

        # Down4
        self.down_4 = MaxPooling2D(2, 2)
        self.conv_bn_relu4 = ConvBNReLU(filters, activation, kernel=1)
        # TODO: why are we using kernel = 1 here?
        self.deconv_4_1 = ConvTranspose2D(filters, filters)
        self.deconv_4_2 = ConvTranspose2D(filters, filters)

        # output
        self.out = Conv2DandReLU(filters * 4, filters)

    def forward(self, R, I_att):
        """
        R : Reflectance
        I_att : Illumination attention
        """
        R_att = R * I_att # WARN: why are we multiplying R and I_att?

        # Down 1
        msia_1 = self.conv_bn_relu1(R_att)

        # Down 2
        down_2 = self.down_2(R_att)
        conv_bn_relu_2 = self.conv_bn_relu2(down_2)
        msia_2 = self.deconv_2(conv_bn_relu_2)

        # Down 4
        down_4 = self.down_4(down_2)
        conv_bn_relu_4 = self.conv_bn_relu4(down_4)
        deconv_4 = self.deconv_4_1(conv_bn_relu_4)
        msia_4 = self.deconv_4_2(deconv_4)

        # Concat
        concat = torch.cat([R, msia_1, msia_2, msia_4], dim=1)
        # NOTE: Revise this part

        out = self.out(concat)
        return out


class ConvBNReLU(nn.Module):
    """
    Convolution + BatchNorm + ReLU
    """

    def __init__(self, channels, activation='lrelu', kernel=3):
        super().__init__()
        self.activation_layer = nn.LeakyReLU(inplace=True) if activation == 'lrelu' else nn.ReLU(inplace=True)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel, padding=kernel // 2),  # TODO: padding
            nn.BatchNorm2d(channels, momentum=0.99),
            self.activation_layer,
        )

    def forward(self, x):
        return self.conv_bn_relu(x)


class DoubleConv(nn.Module):
    """
    Double Convolution
    This module performs two convolution operations followed by a ReLU activation function.
    """
    def __init__(self, in_channels, out_channels, activation='lrelu'):
        super().__init__()
        self.doubleconv = nn.Sequential(
            Conv2DandReLU(in_channels, out_channels, activation),
            Conv2DandReLU(out_channels, out_channels, activation),
        )

    def forward(self, x):
        return self.doubleconv(x)


class ResConv(nn.Module):
    """
    Residual Convolution
    This module performs a residual convolution operation.
    In residual convolution, the input is added to the output of the convolution operation.
    """
    def __init__(self, in_channels, out_channels, activation='lrelu'):
        super().__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True) if activation == 'lrelu' else nn.ReLU(inplace=True)
        # NOTE: we have used a different slope value for the LeakyReLU activation function

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.99)
        self.cbam = CBAM(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.99)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu(bn1)
        cbam = self.cbam(relu1)
        conv2 = self.conv2(cbam)
        bn2 = self.bn2(conv2)
        out = x + bn2
        return out


class Conv2DandReLU(nn.Module):
    """
    Convolution + ReLU
    This module performs the downsampling operation
    Kernel size is fixed to 3x3 and stride is 1
    """

    def __init__(self, in_channels, out_channels, activation='lrelu', stride=1):
        super().__init__()
        self.activation_layer = nn.LeakyReLU(inplace=True) if activation == 'lrelu' else nn.ReLU(inplace=True)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            self.activation_layer,
        )

    def forward(self, x):
        return self.conv_relu(x)


class ConvTranspose2D(nn.Module):
    """
    Transposed Convolution + ReLU
    This module performs the upsampling operation
    """

    def __init__(self, in_channels, out_channels, activation='lrelu'):
        super().__init__()
        self.activation_layer = nn.LeakyReLU(inplace=True) if activation == 'lrelu' else nn.ReLU(inplace=True)
        self.deconv_relu = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            # WARN: removed output padding from orig code
            self.activation_layer,
        )

    def forward(self, x):
        return self.deconv_relu(x)


class MaxPooling2D(nn.Module):
    """
    Max Pooling
    This module perform Max Pooling operation, which is used in the downsampling path
    """

    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.maxpool(x)


class AvgPooling2D(nn.Module):
    """
    Average Pooling
    This module perform Average Pooling operation, which is used in the upsampling path
    """

    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.avgpool(x)


class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    This implements a channel attention module that adaptively recalibrates channel-wise feature responses
    by modeling interdependencies between channels.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # TODO: AdaptiveAvgPool2d

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.sharedMLP(self.avg_pool(x))
        max_out = self.sharedMLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    This implements a spatial attention module to perform adaptive spatial feature recalibration.
    """

    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        # TODO: why are input and output channels 2 and 1?
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # TODO: dim=1?
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # TODO: dim=1?
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module

    This module implements a convolutional block attention module that adaptively recalibrates
    channel-wise and spatial-wise feature responses by explicitly modeling interdependencies
    between channels and spatial locations.
    """

    def __init__(self, channels):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x) * x  # TODO: why multiply?
        out = self.spatial_attention(x) * x
        return out


class Concat(nn.Module):
    """
    Concatenation
    This module performs concatenation of two tensors along the channel dimension
    """

    def forward(self, x, y):
        """
        We first calculate the difference in height and width between the two tensors.
        Then we pad the smaller tensor with zeros on all sides so that it has the same height and width as the larger tensor.
        y is always the smaller tensor
        Finally, we concatenate the two tensors along the channel dimension.
        """
        _, _, xH, xW = x.size()
        _, _, yH, yW = y.size()
        diffY = xH - yH
        diffX = xW - yW
        y = F.pad(y, (diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2))
        return torch.cat([x, y], dim=1)
