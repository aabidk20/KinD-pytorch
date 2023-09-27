import torch
from torch import optim
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchsummary import summary

from utils import sample


class BaseTrainer:
    def __init__(self, config, dataloader, criterion, model,
                 dataloader_test=None, extra_model=None):
        self.initialize(config)
        self.dataloader = dataloader
        self.dataloader_test = dataloader_test
        self.loss_fn = criterion
        self.model = model
        self.extra_model = extra_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # NOTE: moved extra model to device later
        if self.extra_model is not None:
            self.extra_model.to(self.device)
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True # speed up training by using cudnn


    def initialize(self, config):
        self.batch_size = config['batch_size']
        self.length = config['length']
        self.epochs = config['epochs']
        self.steps_per_epoch = config['steps_per_epoch']
        self.print_frequency = config['print_frequency']
        self.save_frequency = config['save_frequency']
        self.weights_dir = config['weights_dir']
        self.samples_dir = config['samples_dir']
        self.learning_rate = config['learning_rate']
        self.noDecom = config['noDecom']

    def train(self):
        print(f'Using device: {self.device}')
        summary(self.model, input_size=(3, 48, 48))

        self.model.to(self.device)
        cudnn.benchmark = True
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        try:
            for iter in range(self.epochs):
                epoch_loss = 0
                step = 0
                iter_start_time = time.time()
                for idx, data in enumerate(self.dataloader):
                    input_ = data['input'].to(self.device)
                    target = data['target'].to(self.device)
                    y_pred = self.model(input_)
                    loss = self.loss_fn(y_pred, target)
                    print(f'Epoch:{iter} | Step:{step} | Loss:{loss.item()}')
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1
                    if idx > 0 and idx % self.save_frequency == 0:
                        torch.save(self.model.state_dict(), os.path.join(self.weights_dir, f'iter_{iter}_idx_{idx}.pth'))
                        print('Saved model')
                        self.test(iter, idx, plotImage=True, saveImage=True)
                iter_end_time = time.time()
                print(f'Epoch {iter} finished in {iter_end_time - iter_start_time} seconds')
        except KeyboardInterrupt:
            torch.save(self.model.state_dict(), os.path.join(self.weights_dir, f'iter_{iter}_idx_{idx}_inter.pth'))
            print('Saved model before quit')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

    def test(self, epoch=-1, plot_dir='./images/samples-illum', **kwargs):
        # WARN: no use of plotimage and saveimage params yet
        self.model.eval()
        for R_low_tensor, I_low_tensor, R_high_tensor,I_high_tensor,name in self.dataloader_test:
            I_low = I_low_tensor.to(self.device)
            I_high = I_high_tensor.to(self.device)
            with torch.no_grad():
                ratio_high2low = torch.mean(torch.div((I_low +0.0001), (I_high + 0.0001)))
                ratio_low2high = torch.mean(torch.div((I_high + 0.0001), (I_low + 0.0001)))
                ratio_high2low_map =torch.ones_like(I_low) * ratio_high2low
                ratio_low2high_map = torch.ones_like(I_high) * ratio_low2high

            I_low2highmap = self.model(I_low, ratio_low2high_map)
            I_high2lowmap = self.model(I_high, ratio_high2low_map)

            I_low2high_np = I_low2highmap.detach().cpu().numpy()[0]
            I_high2low_np = I_high2lowmap.detach().cpu().numpy()[0]
            I_low_np = I_low_tensor.numpy()[0]
            I_high_np = I_high_tensor.numpy()[0]

            sample_imgs = np.concatenate((I_low_np, I_high_np, I_high2low_np, I_low2high_np), axis=0)
            filepath = os.path.join(plot_dir, f'{name}_epoch_{epoch}.png')
            split_point = [0, 1, 2, 3, 4]
            sample(sample_imgs, split=split_point, figure_size=(2, 2), # WARN: imported sample from utils.py , there one in utils folder as well
                   img_dim=self.length, path=filepath, num=epoch)




