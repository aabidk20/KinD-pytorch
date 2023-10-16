import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import time
import yaml
import sys
from tqdm import tqdm
from torchvision.utils import make_grid
from torchvision import transforms
from torchsummary import summary
from base_trainer import BaseTrainer
from losses import *
from models import *
from base_parser import BaseParser
from dataloader import *


class Illum_Trainer(BaseTrainer):
    def __init__(self, config, dataloader, criterion, model,
                 dataloader_test=None, decom_net=None):
        super().__init__(config, dataloader, criterion, model, dataloader_test)
        log(f'Using device: {self.device}')
        self.decom_net = decom_net
        self.decom_net.to(self.device)

    def train(self):
        # torch.set_grad_enabled(True)
        self.model.train()
        self.model.to(self.device)
        # print(self.model)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99426)
        try:
            for iter in range(self.epochs):
                # print(self.epochs)
                epoch_loss = 0
                idx = 0
                hook_number = -1
                iter_start_time = time.time()

                if self.noDecom is True:
                    for R_low_tensor, L_low_tensor, R_high_tensor, L_high_tensor, name in tqdm(self.dataloader):
                        optimizer.zero_grad()
                        L_low = L_low_tensor.to(self.device)
                        L_high = L_high_tensor.to(self.device)
                        with torch.no_grad():
                            ratio_high2low = torch.mean(torch.div((L_low + 0.0001), (L_high + 0.0001)))
                            # WARN: should it be like torch.mean(L_low + 0.0001) / torch.mean(L_high + 0.0001) ?

                            ratio_low2high = torch.mean(torch.div((L_high + 0.0001), (L_low + 0.0001)))

                        L_low2high_map = self.model(L_low, ratio_low2high)
                        L_high2low_map = self.model(L_high, ratio_high2low)
                        # NOTE: what's happening here?

                        if idx % self.print_frequency == 0:
                            hook_number = iter
                            loss = self.loss_fn(L_low2high_map, L_high, hook=hook_number) + self.loss_fn(L_high2low_map,
                                                                                                         L_low,
                                                                                                         hook=hook_number)
                            # NOTE: revise this part

                            hook_number = -1
                            if idx % 30 == 0:
                                log(f'Iter: {iter}_{idx} \t average_loss: {loss.item():.6f}')
                                print(ratio_high2low, ratio_low2high)
                            loss.backward()
                            optimizer.step()
                            idx += 1
                else:
                    for I_low_tensor, I_high_tensor, name in tqdm(self.dataloader):
                        optimizer.zero_grad()
                        I_low = I_low_tensor.to(self.device)
                        I_high = I_high_tensor.to(self.device)

                        #WARN: Turned off grad tracking
                        with torch.no_grad():
                            R_low, L_low = self.decom_net(I_low)
                            R_high, L_high = self.decom_net(I_high)

                        bright_low = torch.mean(L_low)
                        bright_high = torch.mean(L_high)
                        ratio_high2low = torch.div(bright_low, bright_high)
                        ratio_low2high = torch.div(bright_high, bright_low)

                        L_low2high_map = self.model(L_low, ratio_low2high)
                        L_high2low_map = self.model(L_high, ratio_high2low)

                        # print(f'{I_low2high_map.requires_grad=}, {I_high2low_map.requires_grad=}')
                        loss = self.loss_fn(L_low2high_map, L_high, hook=hook_number) + \
                               self.loss_fn(L_high2low_map, L_low, hook=hook_number)

                        if idx % 30 == 0:
                            log(f'Epoch: {iter} | Step: {idx} \t average_loss: {loss.item():.6f}')
                            # print(ratio_high2low, ratio_low2high)
                            log(f'ratio_high2low = {ratio_high2low.item()}, ratio_low_2high = {ratio_low2high.item()}')
                        loss.backward()
                        optimizer.step()
                        idx += 1

                if iter % self.print_frequency == 0:
                    self.test(iter, plot_dir='./images/samples-illum')

                if iter % self.save_frequency == 0:
                    torch.save(self.model.state_dict(), f'../../../SIG/Code_Repos/KinD-pytorch/weights/illum_net.pth' )
                    log(f'Checkpoint {iter} saved for illum_net!')

                scheduler.step()
                iter_end_time = time.time()
                log(f'Time taken: {iter_end_time - iter_start_time} seconds\t lr: {scheduler.get_lr()}')
        except KeyboardInterrupt:
            torch.save(self.model.state_dict(), f'./weights/illum_{iter}_inter.pth')
            log('Saved model before quit')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

    @no_grad
    def test(self, epoch=-1, plot_dir='./images/samples-illum'):
        self.model.eval()
        if self.noDecom:
            for R_low_tensor, L_low_tensor, R_high_tensor, L_high_tensor, name in tqdm(self.dataloader_test):
                L_low = L_low_tensor.to(self.device)
                L_high = L_high_tensor.to(self.device)

                ratio_high2low = torch.mean(torch.div((L_low + 0.0001), (L_high + 0.0001)))
                ratio_low2high = torch.mean(torch.div((L_high + 0.0001), (L_low + 0.0001)))
                print(ratio_low2high)

                bright_low = torch.mean(L_low)
                bright_high = torch.ones_like(bright_low) * 0.3 + bright_low * 0.55
                ratio_high2low = torch.div(bright_low, bright_high)
                ratio_low2high = torch.div(bright_high, bright_low)
                print(ratio_low2high)

                L_low2high_map = self.model(L_low, ratio_low2high)
                L_high2low_map = self.model(L_high, ratio_high2low)

                L_low2high_np = L_low2high_map.detach().cpu().numpy()[0]
                L_high2low_np = L_high2low_map.detach().cpu().numpy()[0]
                L_low_np = L_low_tensor.numpy()[0]
                L_high_np = L_high_tensor.numpy()[0]
                sample_imgs = np.concatenate((L_low_np, L_high_np, L_high2low_np, L_low2high_np), axis=0)

                filepath = os.path.join(plot_dir, f'{name[0]}_epoch_{epoch}.png')
                split_point = [0, 1, 2, 3, 4]
                img_dim = L_low_np.shape[1:]
                sample(sample_imgs, split=split_point, figure_size=(2, 2),
                       img_dim=img_dim, path=filepath, num=epoch)
        else:
            for I_low_tensor, I_high_tensor, name in tqdm(self.dataloader_test):
                I_low = I_low_tensor.to(self.device)
                I_high = I_high_tensor.to(self.device)

                R_low, L_low = self.decom_net(I_low)
                R_high, L_high = self.decom_net(I_high)

                bright_low = torch.mean(L_low)
                bright_high = torch.mean(L_high)
                ratio_high2low = torch.div(bright_low, bright_high)
                ratio_low2high = torch.div(bright_high, bright_low)
                print(ratio_low2high)

                L_low2high_map = self.model(L_low, ratio_low2high)
                L_high2low_map = self.model(L_high, ratio_high2low)

                L_low2high_np = L_low2high_map.detach().cpu().numpy()[0]
                L_high2low_np = L_high2low_map.detach().cpu().numpy()[0]
                L_low_np = L_low.detach().cpu().numpy()[0]
                L_high_np = L_high.detach().cpu().numpy()[0]

                sample_imgs = np.concatenate((L_low_np, L_high_np, L_high2low_np, L_low2high_np), axis=0)
                filepath = os.path.join(plot_dir, f'{name[0]}_epoch_{epoch}.png')
                split_point = [0, 1, 2, 3, 4]
                img_dim = L_low_np.shape[1:]
                sample(sample_imgs, split=split_point, figure_size=(2, 2),
                       img_dim=img_dim, path=filepath, num=epoch)


if __name__ == '__main__':
    criterion = IllumLoss()
    decom_net = DecomNet()
    model = IllumNet()

    parser = BaseParser()
    args = parser.parse()
    # args.checkpoint = None

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    weight_dir = config['weights_dir']
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    if config['checkpoint'] is True:
        if config['noDecom'] is False:
            decom_net = load_weights(decom_net, path=config['decom_net_path'])
            log('DecomNet loaded from decom_net.pth')
        # model = load_weights(model, path=config['illum_net_path'])
        # log('IllumNet loaded from illum_net.pth')

    if config['noDecom'] is True:
        # WARN: Different from the original code
        root_path_train = config['root_path_train_decomposed']
        root_path_test = config['root_path_test_decomposed']
        list_path_train = build_LOLDataset_Decom_list_txt(root_path_train)
        list_path_test = build_LOLDataset_Decom_list_txt(root_path_test)
        log("Building LOLDataset (Decom)")

        # NOTE: crop_size is length in config
        dst_train = LOLDataset_Decom(root_path_train, list_path_train,
                               crop_size=config['length'], to_RAM=True)
        dst_test = LOLDataset_Decom(root_path_test, list_path_test,
                              crop_size=config['length'], to_RAM=True, training=False)

        train_loader = DataLoader(dst_train, batch_size=config['batch_size'],
                                  shuffle=True)
        test_loader = DataLoader(dst_test, batch_size=1)
    else:
        root_path_train = config['root_path_train']
        root_path_test = config['root_path_test']
        list_path_train = build_LOLDataset_list_txt(root_path_train)
        list_path_test = build_LOLDataset_list_txt(root_path_test)
        log("Building LOLDataset")

        # NOTE: crop_size is length in config
        dst_train = LOLDataset(root_path_train, list_path_train,
                               crop_size=config['length'], to_RAM=True)
        dst_test = LOLDataset(root_path_test, list_path_test,
                              crop_size=config['length'], to_RAM=True, training=False)

        train_loader = DataLoader(dst_train, batch_size=config['batch_size'],
                                  shuffle=True)
        test_loader = DataLoader(dst_test, batch_size=1)

    trainer = Illum_Trainer(config, train_loader, criterion, model,
                            dataloader_test=test_loader, decom_net=decom_net)

    if args.mode == 'train':
        trainer.train()
    else:
        trainer.test()

#matched
