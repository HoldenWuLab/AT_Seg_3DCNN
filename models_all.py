# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class ACD3DUNet(nn.Module):
    def __init__(self, input_channels, output_classes):
        super(ACD3DUNet, self).__init__()
        
        self.inconv  = inputConv_3D(input_channels,64)
        self.dblock1 = denseCompBlock3D(64,64)
        self.down1   = downPath_3D(64,128)
        self.dblock2 = denseCompBlock3D(128,128)
        self.down2   = downPath_3D(128,256)
        self.dblock3 = denseCompBlock3D(256,256)
        self.down3   = downPath_3D(256,512)
        self.dblock4 = CBAM3D(512, reduction_ratio = 4, pool_types=['avg', 'max'], no_spatial=False)
        
        self.upConcat1 = CompupConcat3D(512,256)
        self.updense1 = denseCompBlock3D(256,256)
        self.upConcat2 = CompupConcat3D(256,128)
        self.updense2 = denseCompBlock3D(128,128)
        self.upConcat3 = CompupConcat3D(128,64)
        self.updense3 = denseCompBlock3D(64,64)
        
        self.final = nn.Sequential(nn.Conv3d(64, 3, kernel_size=1), 
                                   nn.Softmax(dim=1))
        
        
    def forward(self, x):

        downPath1 = self.inconv(x)
        downPath1 = self.dblock1(downPath1)
        downPath2 = self.down1(downPath1)
        downPath2 = self.dblock2(downPath2)
        downPath3 = self.down2(downPath2)
        downPath3 = self.dblock3(downPath3)
        downPath4 = self.down3(downPath3)
        downPath4 = self.dblock4(downPath4)
        
        up1_concat = self.upConcat1(downPath4, downPath3)
        up1_feat = self.updense1(up1_concat)
        up2_concat = self.upConcat2(up1_feat, downPath2)
        up2_feat = self.updense2(up2_concat)
        up3_concat = self.upConcat3(up2_feat, downPath1)
        up3_feat = self.updense3(up3_concat)

        outputs = self.final(up3_feat)
        
        return outputs
    
    
class UNet3D(nn.Module):
    def __init__(self, num_channels=1, num_classes=3):
        super(UNet3D, self).__init__()
        num_feat = [64, 128, 256, 512]
        
        self.down1 = nn.Sequential(Conv3x3_3D(num_channels, num_feat[0]))

        self.down2 = nn.Sequential(nn.MaxPool3d(kernel_size=2),
                                   Conv3x3_3D(num_feat[0], num_feat[1]))
        
        self.down3 = nn.Sequential(nn.MaxPool3d(kernel_size=2),
                                   Conv3x3_3D(num_feat[1], num_feat[2]))

        self.bottom = nn.Sequential(nn.MaxPool3d(kernel_size=2),
                                   Conv3x3_3D(num_feat[2], num_feat[3]))

        self.up1 = upConcat_3D(num_feat[3], num_feat[2])
        self.upconv1 = Conv3x3_3D(num_feat[3], num_feat[2])

        self.up2 = upConcat_3D(num_feat[2], num_feat[1])
        self.upconv2 = Conv3x3_3D(num_feat[2], num_feat[1])

        self.up3 = upConcat_3D(num_feat[1], num_feat[0])
        self.upconv3 = Conv3x3_3D(num_feat[1], num_feat[0])

        self.final = nn.Sequential(nn.Conv3d(num_feat[0],
                                             num_classes,
                                             kernel_size=1),
                                   nn.Softmax(dim=1))
        

        
    def forward(self, inputs, return_features=False):
        down1_feat = self.down1(inputs)
        down2_feat = self.down2(down1_feat)
        down3_feat = self.down3(down2_feat)
        bottom_feat = self.bottom(down3_feat)

        up1_feat = self.up1(bottom_feat, down3_feat)
        up1_feat = self.upconv1(up1_feat)
        up2_feat = self.up2(up1_feat, down2_feat)
        up2_feat = self.upconv2(up2_feat)
        up3_feat = self.up3(up2_feat, down1_feat)
        up3_feat = self.upconv3(up3_feat)

        if return_features:
            outputs = up3_feat
        else:
            outputs = self.final(up3_feat)

        return outputs



class BasicConv3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True,
                 bias=False):
        super(BasicConv3D, self).__init__()
        self.out_channels = out_ch
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_ch, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate3D(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate3D, self).__init__()
        self.gate_channels = gate_channels
        self.pool_types = pool_types
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return x * scale


class ChannelPool3D(nn.Module):
    def forward(self, x):
        # 1 is the ch dimension
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate3D(nn.Module):
    def __init__(self):
        super(SpatialGate3D, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool3D()
        self.spatial = BasicConv3D(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM3D(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM3D, self).__init__()
        self.ChannelGate = ChannelGate3D(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate3D()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class denseCompBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(denseCompBlock3D, self).__init__()
        # dense block with local competetive layers
        self.ch_arr = [0] * 4
        self.ch_arr[0] = in_ch
        self.ch_arr[1] = self.ch_arr[0] + in_ch
        self.ch_arr[2] = self.ch_arr[1] + in_ch
        self.ch_arr[3] = self.ch_arr[2] + in_ch

        self.conv3x3_3D = nn.Sequential(nn.Conv3d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm3d(in_ch),
                                        nn.ReLU())
        self.conv1x1_L1_3D = nn.Sequential(nn.Conv3d(in_ch, in_ch, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm3d(in_ch),
                                           nn.ReLU())
        self.conv1x1_L2_3D = nn.Sequential(nn.Conv3d(in_ch, in_ch, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm3d(in_ch),
                                           nn.ReLU())
        self.conv1x1_L3_3D = nn.Sequential(nn.Conv3d(in_ch, in_ch, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm3d(in_ch),
                                           nn.ReLU())
        self.BN_L1_3D = nn.BatchNorm3d(self.ch_arr[0])
        self.maxout_L1_3D = nn.Sequential(Maxout())

    def forward(self, input_layer):
        # residual connections for Dense U-Net
        layer1 = self.conv1x1_L1_3D(input_layer)
        layer1 = self.conv3x3_3D(layer1)
        input_layer = self.BN_L1_3D(input_layer)

        layer2 = self.maxout_L1_3D(torch.stack((input_layer, layer1), dim=0))
        layer2 = self.conv1x1_L2_3D(layer2)
        layer2 = self.conv3x3_3D(layer2)
        layer2 = self.BN_L1_3D(layer2)

        output_layer = self.maxout_L1_3D(torch.stack((input_layer, layer1, layer2), dim=0))

        return output_layer


class downPath_3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(downPath_3D, self).__init__()
        self.pooling = nn.MaxPool3d(kernel_size=2)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.conv_layer2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm3d(out_ch)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pooling(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_layer2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class inputConv_3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inputConv_3D, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm3d(out_ch)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CompupConcat3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CompupConcat3D, self).__init__()
        self.deconv = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.Conv1_1 = nn.Conv3d(2*out_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs, down_outputs):
        outputs = self.deconv(inputs)
        out = self.Conv1_1(torch.cat([down_outputs, outputs], 1))
        return out

class Conv3x3_3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv3x3_3D, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm3d(out_ch),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm3d(out_ch),
                                   nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class upConcat_3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(upConcat_3D, self).__init__()
        self.deconv = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, inputs, down_outputs):
        outputs = self.deconv(inputs)
        # skip connections in U-Net
        out = torch.cat([down_outputs, outputs], 1)
        return out

class Maxout(nn.Module):
    def __init__(self):
        super(Maxout, self).__init__()

    def forward(self, input_layer):
        # shape input = (layers(2), batch size, channels, height, width)
        output_layer = torch.max(input_layer, dim=0).values 
        return output_layer
    
    



