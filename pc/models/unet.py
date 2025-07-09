import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet_Big(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet_Big, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.upconv4 = self.upconv(1024, 512)
        self.dec4 = self.conv_block(1024, 512)

        self.upconv3 = self.upconv(512, 256)
        self.dec3 = self.conv_block(512, 256)

        self.upconv2 = self.upconv(256, 128)
        self.dec2 = self.conv_block(256, 128)

        self.upconv1 = self.upconv(128, 64)
        self.dec1 = self.conv_block(128, 64)

        # Output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoding path
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))

        # Decoding path
        d4 = self.upconv4(b)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.final_conv(d1))

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck (Reduced depth)
        self.bottleneck = self.conv_block(512, 512)

        # Decoder
        self.upconv4 = self.upconv(512, 512)
        self.dec4 = self.conv_block(1024, 512)

        self.upconv3 = self.upconv(512, 256)
        self.dec3 = self.conv_block(512, 256)

        self.upconv2 = self.upconv(256, 128)
        self.dec2 = self.conv_block(256, 128)

        self.upconv1 = self.upconv(128, 64)
        self.dec1 = self.conv_block(128, 64)

        # Output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Dropout for regularization
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoding path
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))

        # Decoding path
        d4 = self.upconv4(b)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.final_conv(d1))  # Keep sigmoid for image reconstruction


class UNetReduced(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNetReduced, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 256)

        # Decoder
        self.upconv4 = self.upconv(256, 256)
        self.dec4 = self.conv_block(256, 256)

        self.upconv3 = self.upconv(256, 128)
        self.dec3 = self.conv_block(128, 128)

        self.upconv2 = self.upconv(128, 64)
        self.dec2 = self.conv_block(64, 64)

        self.upconv1 = self.upconv(64, 32)
        self.dec1 = self.conv_block(32, 32)

        # Output layer
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),  # Depthwise conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1),  # Pointwise conv
            nn.GroupNorm(num_groups=8, num_channels=out_channels),  # GroupNorm instead of BatchNorm
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoding path
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))

        # Decoding path
        d4 = self.upconv4(b) + e4
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4) + e3
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3) + e2
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2) + e1
        d1 = self.dec1(d1)

        return torch.sigmoid(self.final_conv(d1))


class UNetDilated(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, dilation_rate=2):
        super(UNetDilated, self).__init__()

        self.dilation_rate = dilation_rate

        # Encoder
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 256)

        # Decoder
        self.upconv4 = self.upconv(256, 256)
        self.dec4 = self.conv_block(256, 256)

        self.upconv3 = self.upconv(256, 128)
        self.dec3 = self.conv_block(128, 128)

        self.upconv2 = self.upconv(128, 64)
        self.dec2 = self.conv_block(64, 64)

        self.upconv1 = self.upconv(64, 32)
        self.dec1 = self.conv_block(32, 32)

        # Output layer
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3,
                padding=self.dilation_rate, dilation=self.dilation_rate, groups=in_channels  # Depthwise dilated conv
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),  # Pointwise conv
            nn.GroupNorm(num_groups=8, num_channels=out_channels),  # GroupNorm instead of BatchNorm
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoding path
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))

        # Decoding path
        d4 = self.upconv4(b) + e4
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4) + e3
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3) + e2
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2) + e1
        d1 = self.dec1(d1)

        return torch.sigmoid(self.final_conv(d1))


class UNetDilated1(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, dilation_rate=2):
        super(UNetDilated1, self).__init__()

        self.dilation_rate = dilation_rate

        # Encoder
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 256)

        # Decoder
        self.upconv4 = self.upconv(256, 256)
        self.dec4 = self.conv_block(512, 256)  # Adjusted for concatenation

        self.upconv3 = self.upconv(256, 128)
        self.dec3 = self.conv_block(256, 128)

        self.upconv2 = self.upconv(128, 64)
        self.dec2 = self.conv_block(128, 64)

        self.upconv1 = self.upconv(64, 32)
        self.dec1 = self.conv_block(64, 32)

        # Output layer
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=self.dilation_rate, dilation=self.dilation_rate),  # Standard dilated conv
            nn.GroupNorm(num_groups=8, num_channels=out_channels),  # GroupNorm for stability
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoding path
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))

        # Decoding path (Using Concatenation Instead of Addition)
        d4 = torch.cat((self.upconv4(b), e4), dim=1)
        d4 = self.dec4(d4)

        d3 = torch.cat((self.upconv3(d4), e3), dim=1)
        d3 = self.dec3(d3)

        d2 = torch.cat((self.upconv2(d3), e2), dim=1)
        d2 = self.dec2(d2)

        d1 = torch.cat((self.upconv1(d2), e1), dim=1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.final_conv(d1))


class UNetReduced1(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNetReduced1, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 256)

        # Decoder
        self.upconv4 = self.upconv(256, 256)
        self.dec4 = self.conv_block(512, 256)  # Updated for concatenation

        self.upconv3 = self.upconv(256, 128)
        self.dec3 = self.conv_block(256, 128)

        self.upconv2 = self.upconv(128, 64)
        self.dec2 = self.conv_block(128, 64)

        self.upconv1 = self.upconv(64, 32)
        self.dec1 = self.conv_block(64, 32)

        # Output layer
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # Standard conv
            nn.GroupNorm(num_groups=8, num_channels=out_channels),  # Keeping GroupNorm
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoding path
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))

        # Decoding path (Using Concatenation Instead of Addition)
        d4 = torch.cat((self.upconv4(b), e4), dim=1)
        d4 = self.dec4(d4)

        d3 = torch.cat((self.upconv3(d4), e3), dim=1)
        d3 = self.dec3(d3)

        d2 = torch.cat((self.upconv2(d3), e2), dim=1)
        d2 = self.dec2(d2)

        d1 = torch.cat((self.upconv1(d2), e1), dim=1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.final_conv(d1))  # Instead of tanh


class PartialConv2d1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(PartialConv2d1, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.weight_mask_updater = torch.ones(1, 1, kernel_size, kernel_size)
        self.slide_window_size = kernel_size ** 2
        self.eps = 1e-8

    def forward(self, x, mask):
        with torch.no_grad():
            updated_mask = F.conv2d(mask, self.weight_mask_updater.to(x.device), bias=None, stride=1, padding=1)
            updated_mask = torch.clamp(updated_mask, 0, 1)

        mask_ratio = self.slide_window_size / (updated_mask + self.eps)
        mask_ratio = mask_ratio * updated_mask  # Zero out invalid locations

        x = self.conv(x * mask)  # Apply convolution only on valid regions
        x = x * mask_ratio  # Normalize

        return x, updated_mask


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output

class UNetPartialConv(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNetPartialConv, self).__init__()

        # Encoder with Partial Convolutions
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 256)

        # Decoder
        self.upconv4 = self.upconv(256, 256)
        self.dec4 = self.conv_block(256, 256)

        self.upconv3 = self.upconv(256, 128)
        self.dec3 = self.conv_block(128, 128)

        self.upconv2 = self.upconv(128, 64)
        self.dec2 = self.conv_block(64, 64)

        self.upconv1 = self.upconv(64, 32)
        self.dec1 = self.conv_block(32, 32)

        # Output layer
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return PartialConv2d(in_channels, out_channels, kernel_size=1, return_mask=True)

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, mask):
        # Encoding path
        e1, m1 = self.enc1(x, mask)
        e2, m2 = self.enc2(F.max_pool2d(e1, 2), F.max_pool2d(m1, 2))
        e3, m3 = self.enc3(F.max_pool2d(e2, 2), F.max_pool2d(m2, 2))
        e4, m4 = self.enc4(F.max_pool2d(e3, 2), F.max_pool2d(m3, 2))

        # Bottleneck
        b, mb = self.bottleneck(F.max_pool2d(e4, 2), F.max_pool2d(m4, 2))

        # Decoding path
        d4 = self.upconv4(b) + e4
        d4, md4 = self.dec4(d4, m4)

        d3 = self.upconv3(d4) + e3
        d3, md3 = self.dec3(d3, m3)

        d2 = self.upconv2(d3) + e2
        d2, md2 = self.dec2(d2, m2)

        d1 = self.upconv1(d2) + e1
        d1, md1 = self.dec1(d1, m1)

        return torch.sigmoid(self.final_conv(d1)), md1  # Output mask


class UNetPartialConv2(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNetPartialConv2, self).__init__()

        # Encoder with Partial Convolutions
        self.enc1 = self.conv_block( in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 256)

        # Decoder
        self.upconv4 = self.upconv(256, 256)
        self.dec4 = self.conv_block(256, 256)

        self.upconv3 = self.upconv(256, 128)
        self.dec3 = self.conv_block(128, 128)

        self.upconv2 = self.upconv(128, 64)
        self.dec2 = self.conv_block(64, 64)

        self.upconv1 = self.upconv(64, 32)
        self.dec1 = self.conv_block(32, 32)

        # Output layer
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """ Returns a block with PartialConv2d followed by ReLU activation """
        return PartialConvBlock(in_channels, out_channels)

    def upconv(self, in_channels, out_channels):
        """ Returns an upconvolution layer """
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, mask):
        # Encoding path
        e1, m1 = self.enc1(x, mask)
        e2, m2 = self.enc2(F.max_pool2d(e1, 2), F.max_pool2d(m1, 2))
        e3, m3 = self.enc3(F.max_pool2d(e2, 2), F.max_pool2d(m2, 2))
        e4, m4 = self.enc4(F.max_pool2d(e3, 2), F.max_pool2d(m3, 2))

        # Bottleneck
        b, mb = self.bottleneck(F.max_pool2d(e4, 2), F.max_pool2d(m4, 2))

        # Decoding path
        d4 = self.upconv4(b) + e4
        d4, md4 = self.dec4(d4, m4)

        d3 = self.upconv3(d4) + e3
        d3, md3 = self.dec3(d3, m3)

        d2 = self.upconv2(d3) + e2
        d2, md2 = self.dec2(d2, m2)

        d1 = self.upconv1(d2) + e1
        d1, md1 = self.dec1(d1, m1)

        return torch.sigmoid(self.final_conv(d1)), md1  # Output with final mask


class PartialConvBlock(nn.Module):
    """
    A wrapper for PartialConv2d + ReLU that supports both input and mask.
    """

    def __init__(self, in_channels, out_channels):
        super(PartialConvBlock, self).__init__()
        self.pconv = PartialConv2d(in_channels, out_channels, kernel_size=3, padding=1, return_mask=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask):
        x, mask = self.pconv(x, mask)
        x = self.relu(x)
        return x, mask



class ResidualUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ResidualUNet, self).__init__()

        # Encoder
        self.enc1 = self.residual_block(in_channels, 32)
        self.enc2 = self.residual_block(32, 64)
        self.enc3 = self.residual_block(64, 128)
        self.enc4 = self.residual_block(128, 256)

        # Bottleneck
        self.bottleneck = self.residual_block(256, 256)

        # Decoder
        self.upconv4 = self.upconv(256, 256)
        self.dec4 = self.residual_block(256, 256)

        self.upconv3 = self.upconv(256, 128)
        self.dec3 = self.residual_block(128, 128)

        self.upconv2 = self.upconv(128, 64)
        self.dec2 = self.residual_block(64, 64)

        self.upconv1 = self.upconv(64, 32)
        self.dec1 = self.residual_block(32, 32)

        # Output layer
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def residual_block(self, in_channels, out_channels):
        """Residual Block using Depthwise Separable Conv"""
        return ResidualConvBlock(in_channels, out_channels)

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoding path
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))

        # Decoding path with residual connections
        d4 = self.upconv4(b) + e4
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4) + e3
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3) + e2
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2) + e1
        d1 = self.dec1(d1)

        return torch.sigmoid(self.final_conv(d1))


class ResidualConvBlock(nn.Module):
    """Residual Convolution Block with Depthwise Separable Convolution"""
    def __init__(self, in_channels, out_channels):
        super(ResidualConvBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),  # Depthwise Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1),  # Pointwise Conv
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
        )

        # 1x1 Conv for skip connection (if needed)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.residual(x)  # Skip connection
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual  # Add skip connection
        return self.relu(x)


class ResBlock(nn.Module):
    """Residual Block with GroupNorm and Conv layers"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        num_groups = min(32, in_channels)  # Ensure num_groups <= in_channels
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)  # Same fix
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels,
                              kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        return x + identity


class SelfAttention(nn.Module):
    """Self-Attention Block for global context"""

    def __init__(self, channels):
        super().__init__()
        num_groups = min(32, channels)
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(B, C, -1), (q, k, v))

        attn = torch.einsum('bci,bcj->bij', q, k) * (C ** -0.5)
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bij,bcj->bci', attn, v).view(B, C, H, W)
        return self.proj(out) + x


class DownBlock(nn.Module):
    """Downsampling block with ResBlock and optional Self-Attention"""

    def __init__(self, in_channels, out_channels, use_attn=False):
        super().__init__()
        self.res = ResBlock(in_channels, out_channels)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.attn = SelfAttention(out_channels) if use_attn else nn.Identity()

    def forward(self, x):
        x = self.res(x)
        x = self.attn(x)
        return self.downsample(x)


class UpBlock(nn.Module):
    """Upsampling block with ResBlock and optional Self-Attention"""

    def __init__(self, in_channels, out_channels, use_attn=False):
        super().__init__()
        self.res = ResBlock(in_channels, out_channels)
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.attn = SelfAttention(out_channels) if use_attn else nn.Identity()

    def forward(self, x):
        x = self.res(x)
        x = self.upsample(x)
        return self.attn(x)


class UNetSD(nn.Module):
    """U-Net used in Stable Diffusion without text conditioning"""

    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()

        # Encoder
        self.enc1 = DownBlock(in_channels, base_channels)
        self.enc2 = DownBlock(base_channels, base_channels * 2, use_attn=True)
        self.enc3 = DownBlock(base_channels * 2, base_channels * 4, use_attn=True)

        # Bottleneck
        self.bottleneck = ResBlock(base_channels * 4, base_channels * 4)

        # Decoder
        self.dec3 = UpBlock(base_channels * 4, base_channels * 2, use_attn=True)
        self.dec2 = UpBlock(base_channels * 2, base_channels, use_attn=True)
        self.dec1 = UpBlock(base_channels, base_channels)

        # Output
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        # Bottleneck
        x = self.bottleneck(x3)

        # Decoder path
        x = self.dec3(x + x3)
        x = self.dec2(x + x2)
        x = self.dec1(x + x1)

        # Output
        return self.final_conv(x)

























