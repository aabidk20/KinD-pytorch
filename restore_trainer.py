import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import yaml
import sys
from tqdm import tqdm
from torch.optim import lr_scheduler
from torchvision.utils import make_grid
from torchvision import transforms
from torchsummary import summary
from base_trainer import BaseTrainer
from losses import *
from models import *
from base_parser import BaseParser
from dataloader import *


class RestoreTrainer(BaseTrainer):
    """
    RestoreTrainer class
    This class is used to train the RestoreNet model.
    """

    def __init__(self, config, dataloader, criterion,
                 model, dataloader_test=None, decom_net=None):
        # NOTE: why are we not passing decom_net here to base class?
        super().__init__(config, dataloader, criterion, model, dataloader_test)
        log(f'Using device: {self.device}')
        self.decom_net = decom_net
        self.decom_net.to(self.device)

    def train(self):
        # summary(self.model, input_size=[(3, 384, 384), (1, 384, 384)])
        summary(self.model, input_size=[(3, 256, 256), (1, 256, 256)])

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.986233)
        try:
            for iter in range(self.epochs):
                epoch_loss = 0
                idx = 0
                hook_number = -1
                iter_start_time = time.time()

                if self.noDecom is True:
                    for R_low_tensor, L_low_tensor, R_high_tensor, L_high_tensor, name in tqdm(self.dataloader):
                        optimizer.zero_grad()
                        L_low = L_low_tensor.to(self.device)
                        R_low = R_low_tensor.to(self.device)
                        R_high = R_high_tensor.to(self.device)
                        R_restore = self.model(R_low, L_low)

                        if idx % self.print_frequency == 0:
                            hook_number = iter
                        loss = self.loss_fn(R_restore, R_high)
                        hook_number = -1

                        if idx % 30 == 0:
                            log(f'Epoch:{iter} | Step:{idx} | Loss:{loss.item()}')
                        loss.backward()
                        optimizer.step()
                        idx += 1
                else:
                    for I_low_tensor, I_high_tensor, name in tqdm(self.dataloader):
                        optimizer.zero_grad()
                        I_low = I_low_tensor.to(self.device)
                        I_high = I_high_tensor.to(self.device)

                        with torch.no_grad():
                            R_low, L_low = self.decom_net(I_low)
                            R_high, L_high = self.decom_net(I_high)

                        R_restore = self.model(R_low, L_low)

                        if idx % self.print_frequency == 0:
                            hook_number = iter
                        loss = self.loss_fn(R_restore, R_high)
                        hook_number = -1
                        if idx % 30 == 0:
                            log(f'Epoch:{iter} | Step:{idx} | Loss:{loss.item()}')
                        loss.backward()
                        optimizer.step()
                        idx += 1

                if iter % self.print_frequency == 0:
                    self.test(iter, plot_dir='./images/samples-restore')

                if iter % self.save_frequency == 0:
                    torch.save(self.model.state_dict(), f'./weights/restore_net.pth')
                    log(f'Checkpoint {iter} saved for restore net!')

                scheduler.step()
                iter_end_time = time.time()
                log(f'Time taken: {iter_end_time - iter_start_time} seconds\t lr: {scheduler.get_lr()}')
        except KeyboardInterrupt:
            torch.save(self.model.state_dict(), f'./weights/restore_{iter}_inter.pth')
            log('Saved model before quit')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

    @no_grad
    def test(self, epoch=-1, plot_dir='./images/samples-restore'):
        self.model.eval()
        if self.noDecom:
            for R_low_tensor, L_low_tensor, R_high_tensor, L_high_tensor, name in tqdm(self.dataloader_test):
                L_low = L_low_tensor.to(self.device)
                R_low = R_low_tensor.to(self.device)
                R_restore = self.model(R_low, L_low)

                R_restore_np = R_restore.detach().cpu().numpy()[0]
                L_low_np = L_low_tensor.numpy()[0]
                R_low_np = R_low_tensor.numpy()[0]
                R_high_np = R_high_tensor.numpy()[0]
                sample_imgs = np.concatenate((L_low_np, R_low_np, R_restore_np, R_high_np), axis=0)

                filepath = os.path.join(plot_dir, f'{name[0]}_epoch_{epoch}.png')
                split_point = [0, 1, 4, 7, 10]
                img_dim = L_low_np.shape[1:]
                sample(sample_imgs, split=split_point, figure_size=(2, 2),
                       img_dim=img_dim, path=filepath, num=epoch)
        else:
            for I_low_tensor, I_high_tensor, name in tqdm(self.dataloader_test):
                I_low = I_low_tensor.to(self.device)
                I_high = I_high_tensor.to(self.device)

                R_low, L_low = self.decom_net(I_low)
                R_high, L_high = self.decom_net(I_high)

                R_restore = self.model(R_low, L_low)

                R_restore_np = R_restore.detach().cpu().numpy()[0]
                L_low_np = L_low.detach().cpu().numpy()[0]
                R_low_np = R_low.detach().cpu().numpy()[0]
                R_high_np = R_high.detach().cpu().numpy()[0]
                sample_imgs = np.concatenate((L_low_np, R_low_np, R_restore_np, R_high_np), axis=0)

                filepath = os.path.join(plot_dir, f'{name[0]}_epoch_{epoch}.png')
                split_point = [0, 1, 4, 7, 10]
                img_dim = L_low_np.shape[1:]
                sample(sample_imgs, split=split_point, figure_size=(2, 2),
                          img_dim=img_dim, path=filepath, num=epoch)

if __name__ == '__main__':
    criterion = RestoreLoss()
    model = RestoreNet_Unet()
    decom_net = DecomNet()
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
            pretrain_decom = torch.load(config['decom_net_path'])# WARN: changed path
            decom_net.load_state_dict(pretrain_decom)
            log(f'DecomNet loaded from {config["decom_net_path"]}')
        # pretrain = torch.load(config['restore_net_path'])# WARN: changed path
        # model.load_state_dict(pretrain)
        # log('RestoreNet loaded from restore_net.pth')

    if config['noDecom'] is True:
        # WARN: Different from the original code
        root_path_train = config['root_path_train_decomposed']
        root_path_test = config['root_path_test_decomposed']
        list_path_train = build_LOLDataset_Decom_list_txt(root_path_train)
        list_path_test = build_LOLDataset_Decom_list_txt(root_path_test)
        log("Building LOLDataset")

        # NOTE: crop size is set to length in config
        dst_train = LOLDataset_Decom(root_path_train, list_path_train,
                               crop_size=config['length'], to_RAM=True)
        dst_test = LOLDataset_Decom(root_path_test, list_path_test,
                              crop_size=config['length'], to_RAM=True, training=False)

        train_loader = DataLoader(dst_train, batch_size=config['batch_size'],
                                  shuffle=True)
        test_loader = DataLoader(dst_test, batch_size=1, shuffle=False)
    else:
        root_path_train = config['root_path_train']
        root_path_test = config['root_path_test']
        list_path_train = build_LOLDataset_list_txt(root_path_train)
        list_path_test = build_LOLDataset_list_txt(root_path_test)
        log("Building LOLDataset")

        # NOTE: crop size is set to length in config
        dst_train = LOLDataset(root_path_train, list_path_train,
                               crop_size=config['length'], to_RAM=True)
        dst_test = LOLDataset(root_path_test, list_path_test,
                              crop_size=config['length'], to_RAM=True, training=False)
        train_loader = DataLoader(dst_train, batch_size=config['batch_size'],
                                  shuffle=True)
        test_loader = DataLoader(dst_test, batch_size=1)

    trainer = RestoreTrainer(config, train_loader, criterion, model,
                             dataloader_test=test_loader, decom_net=decom_net)
    if args.mode == 'train':
        trainer.train()
    else:
        trainer.test()
#matched