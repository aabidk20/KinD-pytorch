import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base_layers import *


class DecomNet(nn.Module):
    """
    Decomposition Net Class
    This class defines the architecture of the Decomposition Net.
    """

    def __init__(self, filters=32, activation='lrelu'):
        """
        layers are named as _r1, _r2 for reflectance path and _i1, _i2 for illumination path
        """
        super().__init__()

        self.conv_input = Conv2DandReLU(3, filters)
        # Top path for Reflectance
        self.maxpool_r1 = MaxPooling2D()
        self.conv_and_relu_r1 = Conv2DandReLU(filters, filters * 2)
        self.maxpool_r2 = MaxPooling2D()
        self.conv_and_relu_r2 = Conv2DandReLU(filters * 2, filters * 4)
        self.deconv_r1 = ConvTranspose2D(filters * 4, filters * 2)
        self.concat_r1 = Concat()
        self.conv_and_relu_r3 = Conv2DandReLU(filters * 4, filters * 2)
        self.deconv_r2 = ConvTranspose2D(filters * 2, filters)
        self.concat_r2 = Concat()
        self.conv_and_relu_r4 = Conv2DandReLU(filters * 2, filters)
        self.conv_r5 = nn.Conv2d(filters, 3, kernel_size=3, padding=1) # WARN: padding
        self.sigmoid_r = nn.Sigmoid()

        # Bottom path for Illumination
        self.conv_and_relu_i1 = Conv2DandReLU(filters, filters)
        self.concat_i1 = Concat()
        self.conv_i1 = nn.Conv2d(filters * 2, 1, kernel_size=3, padding=1)
        self.sigmoid_i1 = nn.Sigmoid()

    def forward(self, x):
        decom_conv1 = self.conv_input(x)
        decom_pool1 = self.maxpool_r1(decom_conv1)
        decom_conv2 = self.conv_and_relu_r1(decom_pool1)
        decom_pool2 = self.maxpool_r2(decom_conv2)
        decom_conv3 = self.conv_and_relu_r2(decom_pool2)
        decom_up1 = self.deconv_r1(decom_conv3)
        decom_concat1 = self.concat_r1(decom_up1, decom_conv2)
        decom_conv4 = self.conv_and_relu_r3(decom_concat1)
        decom_up2 = self.deconv_r2(decom_conv4)
        decom_concat2 = self.concat_r2(decom_up2, decom_conv1)
        decom_conv5 = self.conv_and_relu_r4(decom_concat2)
        decom_conv6 = self.conv_r5(decom_conv5)
        decom_R = self.sigmoid_r(decom_conv6)

        decom_i_conv1 = self.conv_and_relu_i1(decom_conv1)
        decom_i_conv2 = self.concat_i1(decom_i_conv1, decom_conv5)
        decom_i_conv3 = self.conv_i1(decom_i_conv2)
        decom_I = self.sigmoid_i1(decom_i_conv3)

        return decom_R, decom_I


class IllumNet(nn.Module):
    """
    Illumination Net Class.
    This class defines the architecture of the Illumination Net.
    """
    def __init__(self, filters=32, activation='lrelu'):
        super().__init__()
        self.concat_1 = Concat()
        self.conv_and_relu_1 = Conv2DandReLU(2, filters)
        self.conv_and_relu_2 = Conv2DandReLU(filters, filters)
        self.conv_and_relu_3 = Conv2DandReLU(filters, filters)
        self.conv_1 = nn.Conv2d(filters, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, I, ratio):
        with torch.no_grad():
            ratio_map = torch.ones_like(I) * ratio
        adjust_concat1 = self.concat_1(I, ratio_map)
        adjust_conv1 = self.conv_and_relu_1(adjust_concat1)
        adjust_conv2 = self.conv_and_relu_2(adjust_conv1)
        adjust_conv3 = self.conv_and_relu_3(adjust_conv2)
        adjust_conv4 = self.conv_1(adjust_conv3)
        adjust_I = self.sigmoid(adjust_conv4)
        # print(f'{adjust_I.requires_grad=}')
        return adjust_I


# NOTE: skipped Restorenet_msia and custom_illum classes

class RestoreNet_Unet(nn.Module):
    """
    RestoreNet Class
    This class defines the architecture of the RestoreNet.
    """
    def __init__(self, filters=32, activation='lrelu'):
        super().__init__()
        self.concat_1 = Concat()
        self.conv_and_relu_1 = Conv2DandReLU(4,filters)
        self.conv_and_relu_2 = Conv2DandReLU(filters, filters)
        self.maxpool_1 = MaxPooling2D()

        self.conv_and_relu_3 = Conv2DandReLU(filters, filters * 2)
        self.conv_and_relu_4 = Conv2DandReLU(filters * 2, filters * 2)
        self.maxpool_2 = MaxPooling2D()

        self.conv_and_relu_5 = Conv2DandReLU(filters * 2, filters * 4)
        self.conv_and_relu_6 = Conv2DandReLU(filters * 4, filters * 4)
        self.maxpool_3 = MaxPooling2D()

        self.conv_and_relu_7 = Conv2DandReLU(filters * 4, filters * 8)
        self.conv_and_relu_8 = Conv2DandReLU(filters * 8, filters * 8)
        self.maxpool_4 = MaxPooling2D()

        self.conv_and_relu_9 = Conv2DandReLU(filters * 8, filters * 16)
        self.conv_and_relu_10 = Conv2DandReLU(filters * 16, filters * 16)
        # WARN: Dropout removed
        self.deconv_1 = ConvTranspose2D(filters * 16, filters * 8)
        self.concat_2 = Concat()

        self.conv_and_relu_11 = Conv2DandReLU(filters * 16, filters * 8)
        self.conv_and_relu_12 = Conv2DandReLU(filters * 8, filters * 8)
        self.deconv_2 = ConvTranspose2D(filters * 8, filters * 4)
        self.concat_3 = Concat()

        self.conv_and_relu_13 = Conv2DandReLU(filters * 8, filters * 4)
        self.conv_and_relu_14 = Conv2DandReLU(filters * 4, filters * 4)
        self.deconv_3 = ConvTranspose2D(filters * 4, filters * 2)
        self.concat_4 = Concat()

        self.conv_and_relu_15 = Conv2DandReLU(filters * 4, filters * 2)
        self.conv_and_relu_16 = Conv2DandReLU(filters * 2, filters * 2)
        self.deconv_4 = ConvTranspose2D(filters * 2, filters)
        self.concat_5 = Concat()

        self.conv_and_relu_17 = Conv2DandReLU(filters * 2, filters)
        self.conv_and_relu_18 = Conv2DandReLU(filters, filters)
        # WARN: Paper has 256 in output, but implementation has 32 here. We are using 256

        self.conv_1 = nn.Conv2d(filters, 3, kernel_size=3, stride=1, padding=1) # WARN: padding
        self.sigmoid = nn.Sigmoid()

    def forward(self, R, I):
        """
        R: output decom_conv_5 of DecomNet  #WARN should it be decom_conv_6 in paper?
        I: output decom_i_conv3 of DecomNet
        # WARN: in what order should they be concatenated?
        """

        # x = torch.cat([R, I], dim=1)
        # re_concat1 = self.concat_1(x)
        #WARN: replacing above 2 lines with 1 below
        # print(f'{R.shape=}, {I.shape=}')
        re_concat1 = self.concat_1(R, I)
        # print(re_concat1.shape)
        re_conv1_1 = self.conv_and_relu_1(re_concat1)
        re_conv1_2 = self.conv_and_relu_2(re_conv1_1)
        re_pool1 = self.maxpool_1(re_conv1_2)

        re_conv2_1 = self.conv_and_relu_3(re_pool1)
        re_conv2_2 = self.conv_and_relu_4(re_conv2_1)
        re_pool2 = self.maxpool_2(re_conv2_2)

        re_conv3_1 = self.conv_and_relu_5(re_pool2)
        re_conv3_2 = self.conv_and_relu_6(re_conv3_1)
        re_pool3 = self.maxpool_3(re_conv3_2)

        re_conv4_1 = self.conv_and_relu_7(re_pool3)
        re_conv4_2 = self.conv_and_relu_8(re_conv4_1)
        re_pool4 = self.maxpool_4(re_conv4_2)

        re_conv5_1 = self.conv_and_relu_9(re_pool4)
        re_conv5_2 = self.conv_and_relu_10(re_conv5_1)
        re_up1 = self.deconv_1(re_conv5_2)
        re_concat2 = self.concat_2(re_conv4_2, re_up1)

        re_conv6_1 = self.conv_and_relu_11(re_concat2)
        re_conv6_2 = self.conv_and_relu_12(re_conv6_1)
        re_up2 = self.deconv_2(re_conv6_2)
        re_concat3 = self.concat_3(re_conv3_2, re_up2)

        re_conv7_1 = self.conv_and_relu_13(re_concat3)
        re_conv7_2 = self.conv_and_relu_14(re_conv7_1)
        re_up3 = self.deconv_3(re_conv7_2)
        re_concat4 = self.concat_4(re_conv2_2, re_up3)

        re_conv8_1 = self.conv_and_relu_15(re_concat4)
        re_conv8_2 = self.conv_and_relu_16(re_conv8_1)
        re_up4 = self.deconv_4(re_conv8_2)
        re_concat5 = self.concat_5(re_conv1_2, re_up4)

        re_conv9_1 = self.conv_and_relu_17(re_concat5)
        re_conv9_2 = self.conv_and_relu_18(re_conv9_1)

        re_conv10 = self.conv_1(re_conv9_2)
        re_R = self.sigmoid(re_conv10)
        return re_R


class KinD_noDecom(nn.Module):
    """
    The entire network for KinD without decomposition net
    """
    def __init__(self, filters=32, activation='lrelu'):
        super().__init__()
        self.restore_net = RestoreNet_Unet(filters, activation) # WARN: params missing in original
        self.illum_net = IllumNet(filters, activation)

    def forward(self, R, I, ratio):
        I_final = self.illum_net(I, ratio)
        R_final = self.restore_net(R, I) # WARN: should pass I or I_final?, (mostly I)
        I_final_3 = torch.cat([I_final, I_final, I_final], dim=1)  # WARN: why dim=1?
        out = I_final_3 * R_final
        return R_final, I_final, out


class KinD(nn.Module):
    def __init__(self, filters=32, activation='lrelu'):
        super().__init__()

        # WARN: DIFF FROM ORIGINAL CODE
        self.decom_net = DecomNet(filters, activation)
        self.restore_net = RestoreNet_Unet(filters, activation)
        self.illum_net = IllumNet(filters, activation)
        self.KinD_noDecom = KinD_noDecom(filters, activation)
        self.KinD_noDecom.restore_net = self.restore_net # NOTE: overwrite restore_net and illum_net?
        self.KinD_noDecom.illum_net = self.illum_net

    def forward(self, L, ratio):
        R, I = self.decom_net(L)
        R_final, I_final, out = self.KinD_noDecom(R, I, ratio)
        return R_final, I_final, out


class KinD_plus(nn.Module):
    def __init__(self, filters=32, activation='lrelu'):
        super().__init__()
        self.decom_net = DecomNet(filters, activation)
        self.restore_net = RestoreNet_Unet(filters, activation)
        self.illum_net = IllumNet(filters, activation)

    def forward(self, L, ratio):
        R, I = self.decom_net(L)
        I_final = self.illum_net(I, ratio)
        R_final = self.restore_net(R, I) # WARN: should pass I or I_final?
        I_final_3 = torch.cat([I_final, I_final, I_final], dim=1)
        out = I_final_3 * R_final
        return R_final, I_final, out

























