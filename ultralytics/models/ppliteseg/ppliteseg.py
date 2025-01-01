from typing import Union

import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from layers import ConvBN, ConvBNRelu, UAFM, UAFM_SpAtten, init_params
from stdcnet import STDCNet


def load_entire_model(model, pretrained):
    pass


class PPLiteSegHead(nn.Module):
    """
    The head of PPLiteSeg.

    Args:
        backbone_out_chs (List(Tensor)): The channels of output tensors in the backbone.
        arm_out_chs (List(int)): The out channels of each arm module.
        cm_bin_sizes (List(int)): The bin size of context module.
        cm_out_ch (int): The output channel of the last context module.
        arm_type (str): The type of attention refinement module.
        resize_mode (str): The resize mode for the upsampling operation in decoder.
    """

    def __init__(self, backbone_out_chs, arm_out_chs, cm_bin_sizes, cm_out_ch,
                 arm_type, resize_mode):
        super().__init__()

        self.cm = PPContextModule(backbone_out_chs[-1], cm_out_ch, cm_out_ch,
                                  cm_bin_sizes)

        # assert hasattr(layers, arm_type), \
        #     "Not support arm_type ({})".format(arm_type)
        # arm_class = eval("layers." + arm_type)
        arm_class = eval(arm_type)

        self.arm_list = nn.Sequential()  # [..., arm8, arm16, arm32]
        for i in range(len(backbone_out_chs)):
            low_chs = backbone_out_chs[i]
            high_ch = cm_out_ch if i == len(
                backbone_out_chs) - 1 else arm_out_chs[i + 1]
            out_ch = arm_out_chs[i]
            arm = arm_class(low_chs,
                            high_ch,
                            out_ch,
                            ksize=3,
                            resize_mode=resize_mode)
            self.arm_list.append(arm)

    def forward(self, in_feat_list):
        """
        Args:
            in_feat_list (List(Tensor)): Such as [x2, x4, x8, x16, x32].
                x2, x4 and x8 are optional.
        Returns:
            out_feat_list (List(Tensor)): Such as [x2, x4, x8, x16, x32].
                x2, x4 and x8 are optional.
                The length of in_feat_list and out_feat_list are the same.
        """

        high_feat = self.cm(in_feat_list[-1])
        out_feat_list = []

        for i in reversed(range(len(in_feat_list))):
            low_feat = in_feat_list[i]
            arm = self.arm_list[i]
            high_feat = arm(low_feat, high_feat)
            out_feat_list.insert(0, high_feat)

        return out_feat_list


class PPContextModule(nn.Module):
    """
    Simple Context module. Simple Pyramid Pooling Module.

    Args:
        in_channels (int): The number of input channels to pyramid pooling module.
        inter_channels (int): The number of inter channels to pyramid pooling module.
        out_channels (int): The number of output channels after pyramid pooling module.
        bin_sizes (tuple, optional): The out size of pooled feature maps. Default: (1, 3).
        align_corners (bool): An argument of F.interpolate. It should be set to False
            when the output size of feature is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """

    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 bin_sizes,
                 align_corners=False):
        super().__init__()

        self.stages = nn.ModuleList([
            self._make_stage(in_channels, inter_channels, size)
            for size in bin_sizes
        ])

        self.conv_out = ConvBNRelu(in_channels=inter_channels,
                                   out_channels=out_channels,
                                   kernel_size=3,
                                   padding=1)

        self.align_corners = align_corners

    def _make_stage(self, in_channels, out_channels, size):
        prior = nn.AdaptiveAvgPool2d(output_size=size)
        conv = ConvBNRelu(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=1)
        return nn.Sequential(prior, conv)

    def forward(self, input):
        out = None
        input_shape = input.shape[2:]

        for stage in self.stages:
            x = stage(input)
            x = F.interpolate(x,
                              input_shape,
                              mode='bilinear',
                              align_corners=self.align_corners)
            if out is None:
                out = x
            else:
                out += x

        out = self.conv_out(out)
        return out


class SegHead(nn.Module):
    """
    SegClassifier
    """
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = ConvBNRelu(in_chan,
                               mid_chan,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.conv_out = nn.Conv2d(mid_chan,
                                  n_classes,
                                  kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x


class PPLiteSeg(nn.Module):
    """
    The PP_LiteSeg implementation based on PaddlePaddle.

    The original article refers to "Juncai Peng, Yi Liu, Shiyu Tang, Yuying Hao, Lutao Chu,
    Guowei Chen, Zewu Wu, Zeyu Chen, Zhiliang Yu, Yuning Du, Qingqing Dang,Baohua Lai,
    Qiwen Liu, Xiaoguang Hu, Dianhai Yu, Yanjun Ma. PP-LiteSeg: A Superior Real-Time Semantic
    Segmentation Model. https://arxiv.org/abs/2204.02681".

    Args:
        num_classes (int): The number of target classes.
        backbone(nn.Layer): Backbone network, such as stdc1net and resnet18. The backbone must
            has feat_channels, of which the length is 5.
        backbone_indices (List(int), optional): The values indicate the indices of output of backbone.
            Default: [2, 3, 4].
        arm_type (str, optional): The type of attention refinement module. Default: ARM_Add_SpAttenAdd3.
        cm_bin_sizes (List(int), optional): The bin size of context module. Default: [1,2,4].
        cm_out_ch (int, optional): The output channel of the last context module. Default: 128.
        arm_out_chs (List(int), optional): The out channels of each arm module. Default: [64, 96, 128].
        seg_head_inter_chs (List(int), optional): The intermediate channels of segmentation head.
            Default: [64, 64, 64].
        resize_mode (str, optional): The resize mode for the upsampling operation in decoder.
            Default: bilinear.
        pretrained (str, optional): The path or url of pretrained model. Default: None.

    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=[2, 3, 4],
                 arm_type='UAFM_SpAtten',
                 cm_bin_sizes=[1, 2, 4],
                 cm_out_ch=128,
                 arm_out_chs=[64, 96, 128],
                 seg_head_inter_chs=[64, 64, 64],
                 resize_mode='bilinear',
                 pretrained=None):
        super().__init__()

        # backbone
        assert hasattr(backbone, 'feat_channels'), \
            "The backbone should has feat_channels."
        assert len(backbone.feat_channels) >= len(backbone_indices), \
            f"The length of input backbone_indices ({len(backbone_indices)}) should not be" \
            f"greater than the length of feat_channels ({len(backbone.feat_channels)})."
        assert len(backbone.feat_channels) > max(backbone_indices), \
            f"The max value ({max(backbone_indices)}) of backbone_indices should be " \
            f"less than the length of feat_channels ({len(backbone.feat_channels)})."
        self.backbone = backbone

        assert len(backbone_indices) > 1, "The lenght of backbone_indices " \
            "should be greater than 1"
        self.backbone_indices = backbone_indices  # [..., x16_id, x32_id]
        backbone_out_chs = [backbone.feat_channels[i] for i in backbone_indices]

        # head
        if len(arm_out_chs) == 1:
            arm_out_chs = arm_out_chs * len(backbone_indices)
        assert len(arm_out_chs) == len(backbone_indices), "The length of " \
            "arm_out_chs and backbone_indices should be equal"

        self.ppseg_head = PPLiteSegHead(backbone_out_chs, arm_out_chs,
                                        cm_bin_sizes, cm_out_ch, arm_type,
                                        resize_mode)

        if len(seg_head_inter_chs) == 1:
            seg_head_inter_chs = seg_head_inter_chs * len(backbone_indices)
        assert len(seg_head_inter_chs) == len(backbone_indices), "The length of " \
            "seg_head_inter_chs and backbone_indices should be equal"
        self.seg_heads = nn.Sequential()  # [..., head_16, head32]
        for in_ch, mid_ch in zip(arm_out_chs, seg_head_inter_chs):
            self.seg_heads.append(SegHead(in_ch, mid_ch, num_classes))

        # pretrained
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        x_hw = x.shape[2:]

        feats_backbone = self.backbone(x)  # [x2, x4, x8, x16, x32]
        assert len(feats_backbone) >= len(self.backbone_indices), \
            f"The nums of backbone feats ({len(feats_backbone)}) should be greater or " \
            f"equal than the nums of backbone_indices ({len(self.backbone_indices)})"

        feats_selected = [feats_backbone[i] for i in self.backbone_indices]

        feats_head = self.ppseg_head(feats_selected)  # [..., x8, x16, x32]

        if self.training:
            logit_list = []

            for x, seg_head in zip(feats_head, self.seg_heads):
                x = seg_head(x)
                logit_list.append(x)

            logit_list = [
                F.interpolate(x, x_hw, mode='bilinear', align_corners=False)
                for x in logit_list
            ]
        else:
            x = self.seg_heads[0](feats_head[0])
            x = F.interpolate(x, x_hw, mode='bilinear', align_corners=False)
            logit_list = [x]

        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            load_entire_model(self, self.pretrained)
        else:
            init_params(self)


if __name__ == '__main__':
    backbone = STDCNet()
    model = PPLiteSeg(num_classes=4, backbone=backbone)
    # for name, module in model.named_modules():
    #     print(name)
    #     print('======================')
    model.train()
    im = torch.randn([1, 3, 512, 512], dtype=torch.float32)
    for e in range(1):
        out = model(im)
        print(len(out))
        for i, feat in enumerate(out):
            print(f'feat {i} shape: {feat.shape}')

    model.eval()
    with torch.no_grad():
        out = model(im)
        print(len(out))
        for i, feat in enumerate(out):
            print(f'feat {i} shape: {feat.shape}')
    # torch.save(model.state_dict(), 'ppliteseg.pth')
    # print(torch.randint(-9, 9, [2, 4, 3, 3]))



# import math
# import numpy as np

# def weight_copy_torch2paddle(torch_model, paddle_model):
#     torch_dict = torch_model.state_dict()
#     paddle_dict = paddle_model.state_dict()
#
#     filtered_torch_items = [(name, param) for name, param in torch_dict.items() if 'num_batches_tracked' not in name]
#
#     for (_, torch_param), (_, paddle_param) in zip(filtered_torch_items, paddle_dict.items()):
#         numpy_param = torch_param.detach().cpu().numpy()
#         paddle_tensor = paddle.to_tensor(numpy_param)
#         paddle_param.set_value(paddle_tensor)


# class TorchModel(torch.nn.Module):
#     def __init__(self, in_c, out_c, ks=3, s=1, p=0):
#         super(TorchModel, self).__init__()
#         self.conv1 = torch.nn.Conv2d(in_channels=in_c,out_channels=out_c, kernel_size=ks, stride=s, padding=p)
#         self.bn1 = torch.nn.BatchNorm2d(num_features=out_c, eps=0, momentum=0, affine=False)
#
#     def forward(self, x):
#         out1 = self.conv1(x)
#
#         n, c, h, w = out1.shape
#         for j in range(c):
#             print(out1[:, j, :, :])
#             mean = torch.mean(out1[:, j, :, :])
#             std = torch.var(out1[:, j, :, :])
#             print("mean: ", mean, "std: ", std)
#
#             a = float(out1[0, j, 0, 0].to('cpu').data)
#             b = float(out1[1, j, 0, 0].to('cpu').data)
#             c = float(out1[2, j, 0, 0].to('cpu').data)
#             d = float(out1[3, j, 0, 0].to('cpu').data)
#             print(a, b, c, d)
#             print(np.mean([a, b, c, d]))
#             print(np.var([a, b, c, d]))
#
#             m = (a + b + c + d) / 4
#             s = ((a - m) ** 2 + (b - m) ** 2 + (c - m) ** 2 + (d - m) ** 2) / 4
#             a = (a - m) / math.sqrt(s + 0.000000001)
#             b = (b - m) / math.sqrt(s + 0.000000001)
#             c = (c - m) / math.sqrt(s + 0.000000001)
#             d = (d - m) / math.sqrt(s + 0.000000001)
#             print(m, s, math.sqrt(s), a, b, c, d)
#
#         out2 = self.bn1(x)
#         return out2
#

# class PaddleModel(paddle.nn.Layer):
#     def __init__(self, in_c, out_c, ks=3, s=1, p=0):
#         super(PaddleModel, self).__init__()
#         self.conv1 = paddle.nn.Conv2D(in_channels=in_c, out_channels=out_c, kernel_size=ks, stride=s, padding=p)
#         self.bn1 = paddle.nn.BatchNorm2D(num_features=out_c, weight_attr=False, bias_attr=False)
#
#
#     def forward(self, x):
#         out1 = self.conv1(x)
#
#         n, c, h, w = out1.shape
#         for j in range(c):
#             print(out1[:, j, :, :])
#             mean = paddle.mean(out1[:, j, :, :], axis=0)
#             std = paddle.var(out1[:, j, :, :], axis=0)
#             print("mean: ", mean, "std: ", std)
#
#             a = float(out1[0, j, 0, 0].to('cpu').data)
#             b = float(out1[1, j, 0, 0].to('cpu').data)
#             c = float(out1[2, j, 0, 0].to('cpu').data)
#             d = float(out1[3, j, 0, 0].to('cpu').data)
#
#             m = (a + b + c + d) / 4
#             s = ((a - m) ** 2 + (b - m) ** 2 + (c - m) ** 2 + (d - m) ** 2) / 4
#             a = (a - m) / math.sqrt(s + 0.000000001)
#             b = (b - m) / math.sqrt(s + 0.000000001)
#             c = (c - m) / math.sqrt(s + 0.000000001)
#             d = (d - m) / math.sqrt(s + 0.000000001)
#             print(m, s, math.sqrt(s), a, b, c, d)
#
#         out2 = self.bn1(out1)
#         self.bn1.weight.mean().numpy()
#         return out2
#
#     def batch_norm(self, x):
#         n, c, h, w = x.shape
#         for j in range(c):
#             print(x[:, j, :, :])
#             # mean = paddle.mean(x[:, j, :, :], axis=[1, 2])
#             # print(mean)
#             for i in range(n):
#                 x[n, c, 0, 0]


# if __name__ == "__main__":
#
#     # torch.manual_seed(42)
#     np.random.seed(42)
#
#     data = np.array(
#         [
#             [[[0.3745401188473625, 0.9507143064099162, 0.7319939418114051],
#               [0.5986584841970366, 0.15601864044243652, 0.15599452033620265],
#               [0.05808361216819946, 0.8661761457749352, 0.6011150117432088]],
#              [[0.7080725777960455, 0.020584494295802447, 0.9699098521619943],
#               [0.8324426408004217, 0.21233911067827616, 0.18182496720710062],
#               [0.18340450985343382, 0.3042422429595377, 0.5247564316322378]]],
#             [[[0.43194501864211576, 0.2912291401980419, 0.6118528947223795],
#               [0.13949386065204183, 0.29214464853521815, 0.3663618432936917],
#               [0.45606998421703593, 0.7851759613930136, 0.19967378215835974]],
#              [[0.5142344384136116, 0.5924145688620425, 0.046450412719997725],
#               [0.6075448519014384, 0.17052412368729153, 0.06505159298527952],
#               [0.9488855372533332, 0.9656320330745594, 0.8083973481164611]]],
#             [[[0.3046137691733707, 0.09767211400638387, 0.6842330265121569],
#               [0.4401524937396013, 0.12203823484477883, 0.4951769101112702],
#               [0.034388521115218396, 0.9093204020787821, 0.2587799816000169]],
#              [[0.662522284353982, 0.31171107608941095, 0.5200680211778108],
#               [0.5467102793432796, 0.18485445552552704, 0.9695846277645586],
#               [0.7751328233611146, 0.9394989415641891, 0.8948273504276488]]],
#             [[[0.5978999788110851, 0.9218742350231168, 0.0884925020519195],
#               [0.1959828624191452, 0.045227288910538066, 0.32533033076326434],
#               [0.388677289689482, 0.2713490317738959, 0.8287375091519293]],
#              [[0.3567533266935893, 0.28093450968738076, 0.5426960831582485],
#               [0.14092422497476265, 0.8021969807540397, 0.07455064367977082],
#               [0.9868869366005173, 0.7722447692966574, 0.1987156815341724]]]
#         ]
#     )
#
#     # paddle_input = paddle.to_tensor(data, dtype='float32')
#     torch_input = torch.tensor(data, dtype=torch.float32)
#
#     torch_model = TorchModel(data.shape[1], data.shape[1])
#     # paddle_model = PaddleModel(data.shape[1], data.shape[1]*2)
#
#     # weight_copy_torch2paddle(torch_model, paddle_model)
#
#     # paddle_model.eval()
#     torch_model.eval()
#
#     # paddle_out = paddle_model(paddle_input)
#     torch_out = torch_model(torch_input)
#
#     # print(f"model.eval():" )
#     # print(f"Max absolute difference: {np.max(paddle_out.numpy() - torch_out.cpu().detach().numpy())}")
#
#     # paddle_model.train()
#     torch_model.train()
#
#     # paddle_out = paddle_model(paddle_input)
#     torch_out = torch_model(torch_input)
#
#     print(f"model.train():" )
#     # print(f"Max absolute difference: {np.max(paddle_out.numpy() - torch_out.cpu().detach().numpy())}")