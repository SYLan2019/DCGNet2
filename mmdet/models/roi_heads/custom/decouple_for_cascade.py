import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, constant_init, kaiming_init

class NonLocalBlock(nn.Module):
    """
    调用过程
    NONLocalBlock2D(in_channels=32),
    super(NONLocalBlock2D, self).__init__(in_channels,
            inter_channels=inter_channels,
            dimension=2, sub_sample=sub_sample,
            bn_layer=bn_layer)
    """
    def __init__(self,
                 in_channels=256,
                 inter_channels=None,
                 dimension=2,
                 sub_sample=True,
                 bn_layer=True):
        super(NonLocalBlock, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 8
            # 进行压缩得到channel个数
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c,  h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)#[bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)


        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True

class ContextBlock2d(nn.Module):

    def __init__(self, inplanes=256, planes=256, pool='att', fusions=['channel_add'], ratio=8):
        super(ContextBlock2d, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)#context Modeling
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)#softmax操作
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


class MultipleScaleFeature(nn.Module):
    def __init__(self, inter_num=3, in_channels=256):
        super(MultipleScaleFeature, self).__init__()
        self.inter_num = inter_num
        self.in_channels = in_channels
        self.mix_block = self.builder_inter_conv()

    def builder_inter_conv(self):
        tmp_conv = []
        for index in range(self.inter_num):
            tmp_conv.append(
                nn.Sequential(
                    DepthwiseSeparableConvModule(in_channels=self.in_channels, out_channels=self.in_channels,
                                                 kernel_size=3, stride=1, padding=1),
                    nn.ReLU()
                )
            )
        return nn.ModuleList(tmp_conv)

    @auto_fp16()
    def forward(self, x):
        tmp = []
        for index, block in enumerate(self.mix_block):
            input = (x if index == 0 else tmp[-1])
            tmp_out = block(input) + x
            tmp.append(tmp_out)
        out = torch.cat(tmp, dim=1)
        return out

class DynamicSpatialSelect(nn.Module):
    def __init__(self, outchannels=256, head_num=2, cat_num=3):
        super(DynamicSpatialSelect, self).__init__()
        self.conv1 = nn.Conv2d(2, head_num, 5, 1, 2)
        self.act = nn.Sigmoid()
        self.reduce_conv1 = nn.Conv2d(in_channels=outchannels * cat_num, out_channels=outchannels, kernel_size=1)
        self.reduce_conv2 = nn.Conv2d(in_channels=outchannels * cat_num, out_channels=outchannels, kernel_size=1)

    @auto_fp16()
    def forward(self, x):
        max_x = torch.max(x, dim=1, keepdim=True)[0]
        mean_x = torch.mean(x, dim=1, keepdim=True)
        cat_fea = torch.cat([max_x, mean_x], dim=1)
        att = self.act(self.conv1(cat_fea))
        x1 = att[:, 0, ...].unsqueeze(1) * x
        x2 = att[:, 1, ...].unsqueeze(1) * x
        #reduce channel
        x1 = self.reduce_conv1(x1)
        x2 = self.reduce_conv1(x2)
        return x1, x2

class DynamicChannelSelect(nn.Module):
    def __init__(self, in_channels=256, branch_num=3):
        super(DynamicChannelSelect, self).__init__()
        self.inchannels = in_channels
        self.branch_num = branch_num
        input_dim = in_channels * branch_num
        self.adap_max_pool = nn.AdaptiveMaxPool2d(1)
        self.adap_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(in_features=input_dim, out_features=input_dim * 2)
        self.act = nn.ReLU()

    def forward(self, x):
        b, c, h, w = x.shape
        max_fea = self.adap_max_pool(x).view(b, c)
        avg_fea = self.adap_avg_pool(x).view(b, c)
        sum_max_avg_fea = max_fea + avg_fea
        all_att = self.act(self.linear1(sum_max_avg_fea))
        all_att = all_att.reshape(b, 2, -1).permute(1, 0, 2).contiguous()
        reg_att, cls_att = all_att[0], all_att[1]
        reg_att_list = torch.chunk(reg_att, self.branch_num, dim=1)
        cls_att_list = torch.chunk(cls_att, self.branch_num, dim=1)
        reg_att = torch.stack(reg_att_list, dim=1)
        cls_att = torch.stack(cls_att_list, dim=1)
        reg_att = F.softmax(reg_att, dim=1)
        cls_att = F.softmax(cls_att, dim=1)
        fea_list = torch.chunk(x, self.branch_num, dim=1)
        fea = torch.stack(fea_list, dim=1)
        reg_fea = reg_att.unsqueeze(-1).unsqueeze(-1) * fea
        cls_fea = cls_att.unsqueeze(-1).unsqueeze(-1) * fea
        reg_fea = torch.sum(reg_fea, dim=1)
        cls_fea = torch.sum(cls_fea, dim=1)
        return reg_fea, cls_fea

class SpatialAttention(nn.Module):
    def __init__(self, in_channels=256):
        super(SpatialAttention, self).__init__()
        self.mix = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.act = nn.ReLU()

    def forward(self, x):
        max_fea = torch.max(x, dim=1, keepdim=True)[0]
        mean_fea = torch.mean(x, dim=1, keepdim=True)
        cat_fea = torch.cat([max_fea, mean_fea], dim=1)
        spatial_att = self.act(self.mix(cat_fea))
        out = spatial_att * x
        return out

class GlobalRefine(BaseModule):
    def __init__(self, in_channels=256):
        super(GlobalRefine, self).__init__()
        self.conv_q = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv_k = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv_v = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)
        tmp = q @ torch.transpose(k, -2, -1)
        score = F.softmax(tmp, dim=-1)
        out = score @ v
        return out

class WHAttention(nn.Module):
    def __init__(self):
        super(WHAttention, self).__init__()
        self.adap_pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Conv1d(1, 1, 3, 1, 1)
        self.act = nn.ReLU()
        self.linear2 = nn.Conv1d(1, 1, 3, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h_fea = x.clone().permute((0, 2, 1, 3)).contiguous()
        w_fea = x.clone().permute((0, 3, 2, 1)).contiguous()
        h_fea = self.adap_pool(h_fea).view(B, H)
        w_fea = self.adap_pool(w_fea).view(B, W)
        cat_fea = torch.cat([h_fea, w_fea], dim=1).unsqueeze(1)
        out = self.linear2(self.act(self.linear1(cat_fea))).view((B, H + W))
        h_att = out[..., :H]
        w_att = out[..., H:]
        return h_att.unsqueeze(1).unsqueeze(3) * x * w_att.unsqueeze(1).unsqueeze(2)


#for AI-TODv2
# class SpatialAndChannelGlobalEnhance(nn.Module):
#     def __init__(self, inchanels=256):
#         super(SpatialAndChannelGlobalEnhance, self).__init__()
#         self.adap_max = nn.AdaptiveMaxPool2d(1)
#         self.adap_avg = nn.AdaptiveAvgPool2d(1)
#         self.conv1 = nn.Conv2d(2, 1, 1, 1)
#         #self.conv1 = nn.Sequential(
#              #nn.Conv2d(inchanels * 2, inchanels // 2, 1, 1),
#              #nn.ReLU(),
#              #nn.Conv2d(inchanels // 2, inchanels, 1, 1)
#         #)
#         self.v_conv = nn.Conv2d(inchanels, inchanels, 1, 1)
#         # self.mlp1 = nn.Sequential(
#         #      nn.Linear(inchanels, inchanels // 4),
#         #      nn.ReLU(inplace=True),
#         #      nn.Linear(inchanels // 4, inchanels)
#         #  )
#         self.mlp2 = nn.Sequential(
#             nn.Conv2d(inchanels, inchanels // 4, 1),
#             #nn.LayerNorm([inchanels // 4, 1, 1]),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inchanels // 4, inchanels, 1)
#         )
#
#
#     def forward(self, x):
#         b, c ,h, w = x.shape
#         max_fea = self.adap_max(x)
#         avg_fea = self.adap_avg(x)
#         cat_fea = torch.cat([max_fea, avg_fea], dim=-1).permute(0, 3, 1, 2)
#         #cat_fea = torch.cat([max_fea, avg_fea], dim=1)
#         tmp_att = self.conv1(cat_fea).squeeze(-1)
#         #tmp_att = self.conv1(cat_fea).squeeze(-1).squeeze(-1).unsqueeze(1)
#         channel_att = F.softmax(tmp_att, dim=-1)
#         v = self.v_conv(x).view(b, c, -1)
#         cross_channel_fea = (channel_att @ v).view(b, 1, h, w)
#         max_h_fea = torch.max(x, dim=3)[0].transpose(-1, -2)
#         max_w_fea = torch.max(x, dim=2)[0]
#         # cat_fea = torch.cat([max_h_fea, max_w_fea], dim=-1).permute(0, 2, 1).contiguous()
#         # cat_fea = self.mlp1(cat_fea)
#         # h_fea = cat_fea[:, :h, ...]
#         # w_fea = cat_fea[:, h:, ...].transpose(-1, -2)
#         #q = (h_fea @ w_fea).view(b, -1).unsqueeze(-1)
#         q = (max_h_fea @ max_w_fea).view(b, -1).unsqueeze(-1)
#         q = F.softmax(q, dim=1)
#         cross_spatial_fea = (v @ q).unsqueeze(-1)
#         cross_spatial_fea = self.mlp2(cross_spatial_fea)
#         refine_fea = cross_channel_fea * cross_spatial_fea
#         out = refine_fea + x
#         #print(self.act)
#         #out = refine_fea * x + x
#         return out

'''class SpatialAndChannelGlobalEnhance(nn.Module):
    def __init__(self, inchanels=256):
        super(SpatialAndChannelGlobalEnhance, self).__init__()
        self.adap_max = nn.AdaptiveMaxPool2d(1)
        self.adap_avg = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(2, 1, 1, 1)
        #self.conv1 = nn.Sequential(
             #nn.Conv2d(inchanels * 2, inchanels // 2, 1, 1),
             #nn.ReLU(),
             #nn.Conv2d(inchanels // 2, inchanels, 1, 1)
        #)
        self.v_conv = nn.Conv2d(inchanels, inchanels, 1, 1)
        self.conv2 = nn.Linear(inchanels, inchanels//8)
        self.act1 = nn.ReLU()
        #self.conv3 = nn.Linear(inchanels//4, inchanels)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(inchanels, inchanels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(inchanels // 4, inchanels, 1)
        )
        self.act2 = nn.Sigmoid()


    def forward(self, x):
        b, c ,h, w = x.shape
        max_fea = self.adap_max(x)
        avg_fea = self.adap_avg(x)
        cat_fea = torch.cat([max_fea, avg_fea], dim=-1).permute(0, 3, 1, 2)
        #cat_fea = torch.cat([max_fea, avg_fea], dim=1)
        tmp_att = self.conv1(cat_fea).squeeze(-1)
        #tmp_att = self.conv1(cat_fea).squeeze(-1).squeeze(-1).unsqueeze(1)
        channel_att = F.softmax(tmp_att, dim=-1)
        v = self.v_conv(x).view(b, c, -1)
        cross_channel_fea = (channel_att @ v).view(b, 1, h, w)
        max_h_fea = torch.max(x, dim=3)[0]
        max_w_fea = torch.max(x, dim=2)[0]
        cat_fea = torch.cat([max_h_fea, max_w_fea], dim=-1).permute(0, 2, 1).contiguous()
        cat_fea = self.act1(self.conv2(cat_fea))
        h_fea = cat_fea[:, :h, ...]
        w_fea = cat_fea[:, h:, ...].transpose(-1, -2)
        q = (h_fea @ w_fea).view(b, -1).unsqueeze(-1)
        q = F.softmax(q, dim=1)
        cross_spatial_fea = (v @ q).unsqueeze(-1)
        cross_spatial_fea = self.channel_mlp(cross_spatial_fea)
        refine_fea = cross_channel_fea * self.act2(cross_spatial_fea)
        out = refine_fea + x
        return out'''

#for twoDecoupleHead
class GCEMv2(nn.Module):
    def __init__(self, inchanels=256):
        super(GCEMv2, self).__init__()
        self.adap_max = nn.AdaptiveMaxPool2d(1)
        self.adap_avg = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(2, 1, 1, 1)
        self.v_conv = nn.Conv2d(inchanels, inchanels, 1, 1)
        self.mlp2 = nn.Sequential(
            nn.Conv2d(inchanels, inchanels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchanels // 4, inchanels, 1)
        )


    def forward(self, x):
        b, c ,h, w = x.shape
        max_fea = self.adap_max(x)
        avg_fea = self.adap_avg(x)
        cat_fea = torch.cat([max_fea, avg_fea], dim=-1).permute(0, 3, 1, 2)
        tmp_att = self.conv1(cat_fea).squeeze(-1)
        channel_att = F.softmax(tmp_att, dim=-1)
        v = self.v_conv(x).view(b, c, -1)
        cross_channel_fea = (channel_att @ v).view(b, 1, h, w)
        max_h_fea = torch.max(x, dim=3)[0].transpose(-1, -2)
        max_w_fea = torch.max(x, dim=2)[0]
        q = (max_h_fea @ max_w_fea).view(b, -1).unsqueeze(-1)
        q = F.softmax(q, dim=1)
        cross_spatial_fea = (v @ q).unsqueeze(-1)
        cross_spatial_fea = self.mlp2(cross_spatial_fea)
        refine_fea = cross_channel_fea * cross_spatial_fea.sigmoid()
        out = refine_fea + x
        return out

class GCEMv1(nn.Module):
    def __init__(self, inchanels=256):
        super(GCEMv1, self).__init__()
        self.adap_max = nn.AdaptiveMaxPool2d(1)
        self.adap_avg = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(2, 1, 1, 1)
        self.v_conv = nn.Conv2d(inchanels, inchanels, 1, 1)
        self.mlp2 = nn.Sequential(
            nn.Conv2d(inchanels, inchanels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchanels // 4, inchanels, 1)
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(inchanels, 1, 1),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        b, c ,h, w = x.shape
        max_fea = self.adap_max(x)
        avg_fea = self.adap_avg(x)
        cat_fea = torch.cat([max_fea, avg_fea], dim=-1).permute(0, 3, 1, 2)
        tmp_att = self.conv1(cat_fea).squeeze(-1)
        channel_att = F.softmax(tmp_att, dim=-1)
        v = self.v_conv(x).view(b, c, -1)
        cross_channel_fea = (channel_att @ v).view(b, 1, h, w)
        q = self.spatial_att(x)
        q = q.view(b, -1).unsqueeze(-1)
        q = F.softmax(q, dim=1)
        cross_spatial_fea = (v @ q).unsqueeze(-1)
        cross_spatial_fea = self.mlp2(cross_spatial_fea)
        refine_fea = cross_channel_fea * cross_spatial_fea.sigmoid()
        out = refine_fea + x
        return out

    
class Mlp(nn.Module):
    def __init__(self, in_features=256, hidden_features=None, out_features=256, act_layer=nn.GELU, drop=0., group=-1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if group > 0:
            self.fc1 = nn.Conv1d(in_features, hidden_features, 1, groups=group)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        if group > 0:
            self.fc2 = nn.Conv1d(hidden_features, out_features, 1, groups=group)
        else:
            self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.group = group

    def forward(self, x):
        if self.group > 0:
            x = x.permute(0, 2, 1).contiguous()
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if self.group > 0:
            x = x.permute(0, 2, 1).contiguous()
        return x

class DecoupleTaskInteraction(nn.Module):
    def __init__(self, input_dim=256, head_num=3, with_position=False):
        super(DecoupleTaskInteraction, self).__init__()
        self.with_position = with_position
        self.mix = nn.Conv2d(in_channels=head_num * input_dim, out_channels=input_dim, kernel_size=1)
        self.norm = nn.LayerNorm(input_dim)
        self.q_conv1 = nn.Linear(input_dim, input_dim,)
        self.q_conv2 = nn.Linear(input_dim, input_dim)
        self.q_conv3 = nn.Linear(input_dim, input_dim)
        self.k_conv = nn.Linear(input_dim, input_dim)
        self.v_conv = nn.Linear(input_dim, input_dim)
        self.center_mlp = Mlp()
        self.wh_mlp = Mlp()
        self.cls_mlp = Mlp()

    def forward(self, center_fea, wh_fea, cls_fea):
        b, c, h, w = center_fea.shape
        cat_fea = torch.cat([center_fea, wh_fea, cls_fea], dim=1)
        mix_fea = self.mix(cat_fea)
        center_fea = center_fea.view(b, c, -1).permute(0, 2, 1).contiguous()
        wh_fea = wh_fea.view(b, c, -1).permute(0, 2, 1).contiguous()
        cls_fea = cls_fea.view(b, c, -1).permute(0, 2, 1).contiguous()
        mix_fea = mix_fea.view(b, c, -1).permute(0, 2, 1).contiguous()
        center_tmp, wh_tmp, cls_tmp, mix_fea = self.norm(center_fea), self.norm(wh_fea), self.norm(cls_fea), self.norm(mix_fea)
        center_q = self.q_conv1(center_tmp)
        wh_q = self.q_conv2(wh_tmp)
        cls_q = self.q_conv3(cls_tmp)
        k_fea = self.k_conv(mix_fea)
        v_fea = self.v_conv(mix_fea)
        center_score = F.softmax(center_q @ k_fea.transpose(-2, -1), dim=2)
        wh_score = F.softmax(wh_q @ k_fea.transpose(-2, -1), dim=2)
        cls_score = F.softmax(cls_q @ k_fea.transpose(-2, -1), dim=2)
        center_out, wh_out, cls_out = center_score @ v_fea, wh_score @ v_fea, cls_score @ v_fea
        center_fea, wh_fea, cls_fea = self.norm(center_fea + center_out), self.norm(wh_fea + wh_out), self.norm(cls_fea + cls_out)
        center_fea, wh_fea, cls_fea = self.norm(center_fea + self.center_mlp(center_fea)), self.norm(wh_fea + self.wh_mlp(wh_fea)), self.norm(cls_fea + self.cls_mlp(cls_fea))
        center_fea, wh_fea, cls_fea = center_fea.permute(0, 2, 1).contiguous().view(b, c, h, w), wh_fea.permute(0, 2, 1).contiguous().view(b, c, h, w), cls_fea.permute(0, 2, 1).contiguous().view(b, c, h, w)
        return center_fea, wh_fea, cls_fea

class FIMv1(nn.Module):
    def __init__(self, inchannels=256):
        super(FIMv1, self).__init__()
        self.inchannels = inchannels
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mean_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inchannels * 3 * 2, inchannels * 3 // 2, 1, 1)
        self.act1 = nn.Sigmoid()
        self.conv2 = nn.Conv2d(inchannels * 3 // 2, inchannels * 3, 1, 1)

    def forward(self, center_fea, wh_fea, cls_fea):
        cat_fea = torch.cat([center_fea, wh_fea, cls_fea], dim=1)
        max_fea = self.max_pool(cat_fea)
        mean_fea = self.mean_pool(cat_fea)
        max_mean_fea = torch.cat([max_fea, mean_fea],dim=1)
        att = self.conv2(self.act1(self.conv1(max_mean_fea)))
        att_list = torch.chunk(att, 3, 1)
        center_out, wh_out, cls_out = att_list[0] * center_fea + center_fea, att_list[1] * wh_fea + wh_fea, att_list[2] * cls_fea + cls_fea
        return center_out, wh_out, cls_out


class FIMv2(nn.Module):
    def __init__(self, channels):
        super(FIMv2, self).__init__()
        self.channels = channels

        # 通道注意力
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(2 * channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, 2 * channels),
            nn.Sigmoid()
        )

    def forward(self, f_cls, f_reg):
        # 拼接特征
        f_cat = torch.cat([f_cls, f_reg], dim=1)  # [B, 2C, H, W]

        # 通道注意力权重
        w = self.gap(f_cat).squeeze(-1).squeeze(-1)  # [B, 2C]
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1)  # [B, 2C, 1, 1]
        w_cls, w_reg = torch.chunk(w, 2, dim=1)  # [B, C, 1, 1] each

        # 动态加权融合
        f_cls_enhanced = f_cls * w_cls + f_reg * (1 - w_cls)
        f_reg_enhanced = f_reg * w_reg + f_cls * (1 - w_reg)

        # 残差连接
        f_cls_out = f_cls + f_cls_enhanced
        f_reg_out = f_reg + f_reg_enhanced

        return f_cls_out, f_reg_out


