import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_ssim
from dataloader import *
from base_parser import BaseParser
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Sobel = torch.tensor([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
Robert = torch.tensor([[0, 0],
                        [-1, 1]])


def feature_map_hook(*args, path=None):
    feature_maps = []
    for feature in args:
        feature_maps.append(feature)
    feature_all = torch.cat(feature_maps, dim=1)
    fmap = feature_all.detach().cpu().numpy()[0]
    fmap = np.array(fmap)
    fshape = fmap.shape
    num = fshape[0]
    shape = fshape[1:]
    sample(fmap, figure_size=(2, num // 2), img_dim=shape, path=path)
    return fmap

def gradient(maps, direction, device=device, kernel='sobel'):
    channels = maps.size()[1]
    if kernel == 'robert':
        smooth_kernel_x = Robert.expand(channels, channels, 2, 2).to(device)
        maps = F.pad(maps,(0, 0, 1, 1))
    elif kernel == 'sobel':
        smooth_kernel_x = Sobel.expand(channels, channels, 3, 3).to(device)
        maps = F.pad(maps,(1, 1, 1, 1))
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)
    if direction == 'x':
        kernel = smooth_kernel_x.type(torch.float32)
    elif direction == 'y':
        kernel = smooth_kernel_y.type(torch.float32)
    kernel = kernel.to(device)
    gradient_orig = torch.abs(F.conv2d(maps, weight=kernel, padding=0))
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 1e-5))
    return grad_norm


def gradient_no_abs(maps, direction, device=device, kernel='sobel'):
    channels = maps.size()[1]
    if kernel == 'robert':
        smooth_kernel_x = Robert.expand(channels, channels, 2, 2)
        maps = F.pad(maps, (0, 0, 1, 1))
    elif kernel == 'sobel':
        smooth_kernel_x = Sobel.expand(channels, channels, 3, 3)
        maps = F.pad(maps, (1, 1, 1, 1))
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)
    if direction == "x":
        kernel = smooth_kernel_x.type(torch.float32)
    elif direction == "y":
        kernel = smooth_kernel_y.type(torch.float32)
    kernel = kernel.to(device=device)
    # kernel size is (2, 2) so need pad bottom and right side
    gradient_orig = torch.abs(F.conv2d(maps, weight=kernel, padding=0))
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm


class DecomLoss(nn.Module):
    def __init__(self):
        super(DecomLoss, self).__init__()

    def reflectance_similarity(self, R_low, R_high):
        return torch.mean(torch.abs(R_low - R_high))

    # WARN: shouldn't this function use low and high grads instead of x and y grads?
    def illumination_smoothness(self, I, L, name='low', hook=-1):
        L_gray = 0.299 * L[:, 0, :, :] + 0.587 * L[:, 1, :, :] + 0.114 * L[:, 2, :, :]
        L_gray = L_gray.unsqueeze(dim=1)
        I_gradient_x = gradient(I, "x")
        L_gradient_x = gradient(L_gray, "x")
        epsilon = 0.01 * torch.ones_like(L_gradient_x) # NOTE: maybe epsilon is not needed as a tensor
        Denominator_x = torch.max(L_gradient_x, epsilon)
        x_loss = torch.abs(torch.div(I_gradient_x, Denominator_x)) # WARN: this line seems fishy
        I_gradient_y = gradient(I, 'y')
        L_gradient_y = gradient(L_gray, 'y')
        Denominator_y = torch.max(L_gradient_y, epsilon)
        y_loss = torch.abs(torch.div(I_gradient_y, Denominator_y)) # WARN: this line seems fishy
        mut_loss = torch.mean(x_loss + y_loss)
        if hook > -1:
            feature_map_hook(I, L_gray, epsilon, I_gradient_x+I_gradient_y,
                             Denominator_x+Denominator_y, x_loss+y_loss,
                             path=f'./images/samples-features/ilux_smooth_{name}_epoch{hook}.png')
        return mut_loss

    def mutual_consistency(self, I_low, I_high, hook=-1):
        low_gradient_x = gradient(I_low, 'x')
        high_gradient_x = gradient(I_high, 'x')
        M_gradient_x = low_gradient_x + high_gradient_x
        x_loss = M_gradient_x * torch.exp(-10 * M_gradient_x) #NOTE: c is hardcoded as 10
        # NOTE: notice that setting c to zero will set x_loss = M_gradient_x, which is same as L1 loss

        low_gradient_y = gradient(I_low, 'y')
        high_gradient_y = gradient(I_high, 'y')
        M_gradient_y = low_gradient_y + high_gradient_y
        y_loss = M_gradient_y * torch.exp(-10 * M_gradient_y) #NOTE: c is hardcoded as 10
        mutual_loss = torch.mean(x_loss + y_loss)
        if hook > -1:
            feature_map_hook(I_low, I_high, low_gradient_x+low_gradient_y,
                             high_gradient_x+high_gradient_y, M_gradient_x+M_gradient_y,
                             x_loss+y_loss,
                             path=f'./images/samples-features/mutual_consist_epoch{hook}.png')
        return mutual_loss

    # WARN: this also does not match paper
    def reconstruction_error(self, R_low, R_high, I_low_3, I_high_3, L_low, L_high):
        # print(f'{type(R_low)=}, {type(I_low_3)=}, {type(L_low)=}')
        # print(f'{type(R_high)=}, {type(I_high_3)=}, {type(L_high)=}')
        # print(I_low_3[0].shape)
        #WARN: I_low_3 is a tuple so using a hack
        I_low_3 = I_low_3[0]
        recon_loss_low = torch.mean(torch.abs(R_low * I_low_3 - L_low))
        recon_loss_high = torch.mean(torch.abs(R_high * I_high_3 - L_high))
        recon_loss = recon_loss_low + recon_loss_high
        return recon_loss

    def forward(self, R_low, R_high, I_low, I_high, L_low, L_high, hook=-1):
        I_low_3 = torch.cat([I_low, I_low, I_low], dim=1), # NOTE: explore torch.expand here
        I_high_3 = torch.cat([I_high, I_high, I_high], dim=1)
        recon_loss = self.reconstruction_error(R_low, R_high, I_low_3, I_high_3, L_low, L_high)
        equal_R_loss = self.reflectance_similarity(R_low, R_high)
        i_mutual_loss = self.mutual_consistency(I_low, I_high, hook=hook)
        ilux_smooth_loss = self.illumination_smoothness(I_low, L_low, hook=hook) + \
                            self.illumination_smoothness(I_high, L_high, name='high', hook=hook)

        # WARN: DIFF FROM ORIGINAL CODE, but matches paper
        decom_loss = recon_loss + 0.01 * equal_R_loss + 0.08 * ilux_smooth_loss + 0.01 * i_mutual_loss
        return decom_loss


# WARN: this does not match paper, paper has MSE loss
class IllumLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def grad_loss(self, low, high, hook=-1):
        x_loss = F.l1_loss(gradient_no_abs(low, 'x'), gradient_no_abs(high, 'x'))
        y_loss = F.l1_loss(gradient_no_abs(low, 'y'), gradient_no_abs(high, 'y'))
        grad_loss_all = x_loss + y_loss
        return grad_loss_all

    def forward(self, I_low, I_high, hook=-1):
        loss_grad = self.grad_loss(I_low, I_high, hook=hook)
        loss_recon = F.l1_loss(I_low, I_high)
        loss_adjust = loss_recon + loss_grad
        return loss_adjust


class RestoreLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim_loss = pytorch_ssim.SSIM()

    def grad_loss(self, low, high, hook=-1):
        x_loss = F.mse_loss(gradient_no_abs(low, 'x'), gradient_no_abs(high, 'x'))
        y_loss = F.mse_loss(gradient_no_abs(low, 'y'), gradient_no_abs(high, 'y'))
        grad_loss_all = x_loss + y_loss
        return grad_loss_all

    def forward(self, R_low, R_high, hook=-1):
        loss_recon = F.l1_loss(R_low, R_high) # WARN: this is different from paper
        loss_ssim = 1 - self.ssim_loss(R_low, R_high)
        loss_restore = loss_recon + loss_ssim
        # WARN: where's the gradient loss?
        return loss_restore


if __name__ == '__main__':
    import argparse
    from dataloader import *
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt

    #NOTE: no parser here in orig code
    parser = BaseParser()
    args = parser.parse()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    root_path_train = config['root_path_train']
    list_path_train = build_LOLDataset_list_txt(root_path_train)
    batch_size = 1 # WARN: hardcoded value
    log("Building LOLDataset")

    dst_test = LOLDataset(root_path_train, list_path_train,
                            to_RAM=True, training=False) # WARN: crop_size is not set
    testloader = DataLoader(dst_test, batch_size=batch_size)
    for i, data in enumerate(testloader):
        L_low, L_high, name = data
        L_gradient_x = gradient_no_abs(L_high, 'x', device=device, kernel='sobel')
        epsilon = 0.01 * torch.ones_like(L_gradient_x)
        Denominator_x = torch.max(L_gradient_x, epsilon)
        imgs = Denominator_x
        img = imgs[1].numpy()
        sample(img, figure_size=(1, 1), img_dim=400)
#MATCHED