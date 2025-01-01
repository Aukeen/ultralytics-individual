import torch
import torch.nn as nn
from torch.nn import init, functional as F


def init_params(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)


class ConvBN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=None,
                 bias=False,
                 momentum=0.9,
                 **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2 if padding is None else padding,
            bias=bias, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=momentum)  # 0.1

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ConvBNRelu(ConvBN):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=None,
                 bias=False,
                 momentum=0.9,
                 inplace=True,
                 **kwargs):
        super(ConvBNRelu, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            momentum=momentum,
            **kwargs
        )
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        out = self.conv(x)

        size_prods = out.shape[0]
        for i in range(len(out.shape) - 2):
            size_prods *= out.shape[i + 2]
        if size_prods != 1:
            out = self.bn(out)

        out = self.relu(out)
        return out


def avg_max_reduce_channel_helper(x, use_concat=True):
    # Reduce hw by avg and max, only support single input
    assert not isinstance(x, (list, tuple))
    # print("x before mean and max:", x.shape)
    mean_value = torch.mean(x, dim=1, keepdim=True)
    max_value = torch.max(x, dim=1, keepdim=True)[0]
    # mean_value = mean_value.unsqueeze(0)
    # print("mean max:", mean_value.shape, max_value.shape)

    if use_concat:
        res = torch.concat([mean_value, max_value], dim=1)
    else:
        res = [mean_value, max_value]
    return res


def avg_max_reduce_channel(x):
    # Reduce hw by avg and max
    # Return cat([avg_ch_0, max_ch_0, avg_ch_1, max_ch_1, ...])
    if not isinstance(x, (list, tuple)):
        return avg_max_reduce_channel_helper(x)
    elif len(x) == 1:
        return avg_max_reduce_channel_helper(x[0])
    else:
        res = []
        for xi in x:
            res.extend(avg_max_reduce_channel_helper(xi, False))
        return torch.concat(res, dim=1)


class UAFM(nn.Module):
    """
    The base of Unified Attention Fusion Module.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__()

        self.conv_x = ConvBNRelu(x_ch,
                                 y_ch,
                                 kernel_size=ksize,
                                 padding=ksize // 2)
        self.conv_out = ConvBNRelu(y_ch,
                                   out_ch,
                                   kernel_size=3,
                                   padding=1)
        self.resize_mode = resize_mode

    def check(self, x: torch.Tensor, y: torch.Tensor):
        assert x.ndim == 4 and y.ndim == 4
        x_h, x_w = x.shape[2:]
        y_h, y_w = y.shape[2:]
        assert x_h >= y_h and x_w >= y_w

    def prepare(self, x, y):
        x = self.prepare_x(x, y)
        y = self.prepare_y(x, y)
        return x, y

    def prepare_x(self, x, y):
        x = self.conv_x(x)
        return x

    def prepare_y(self, x, y):
        y_up = F.interpolate(y, x.shape[2:], mode=self.resize_mode)
        return y_up

    def fuse(self, x, y):
        out = x + y
        out = self.conv_out(out)
        return out

    def forward(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        self.check(x, y)
        x, y = self.prepare(x, y)
        out = self.fuse(x, y)
        return out


class UAFM_SpAtten(UAFM):
    """
    The UAFM with spatial attention, which uses mean and max values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            ConvBNRelu(4, 2, kernel_size=3, padding=1),
            ConvBN(2, 1, kernel_size=3, padding=1))

        #self._scale.stop_gradient = True

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        # atten = avg_max_reduce_channel([x, y])
        # atten = F.sigmoid(self.conv_xy_atten(atten))
        #
        # out = x * atten + y * (self._scale - atten)
        # out = self.conv_out(out)
        # return out

        atten = avg_max_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


