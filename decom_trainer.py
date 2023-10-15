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
from losses import DecomLoss
from models import DecomNet
from base_parser import BaseParser
from dataloader import *

class DecomTrainer(BaseTrainer):
    """
    DecomTrainer class
    This class is used to train the DecomNet model.
    BaseTrainer params:
        config: config file,
        dataloader: dataloader for training,
        criterion: loss function,
        model: model to train,
        dataloader_test: dataloader for testing,
        extra_model: extra model to train
    """

    # WARN: no base class init called here in original code
    def __init__(self, config, dataloader, criterion, model,
                 dataloader_test=None, extra_model=None):
        super().__init__(config, dataloader, criterion, model, dataloader_test, extra_model)

    def train(self):
        print(f'Using device {self.device}')
        self.model.to(device=self.device)
        summary(self.model, input_size=(3, 48, 48))
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.997)
        try:
            for iter in range(self.epochs):
                epoch_loss = 0
                idx = 0
                hook_number = -1
                iter_start_time = time.time()

                for L_low_tensor, L_high_tensor, name in self.dataloader:
                    L_low = L_low_tensor.to(self.device)
                    L_high = L_high_tensor.to(self.device)
                    R_low, I_low = self.model(L_low)
                    ##############################################
                    # print(f'{R_low.shape=}, {I_low.shape=}, {L_low.shape=}')
                    # print(f'{R_low.dtype=}, {I_low.dtype=}, {L_low.dtype=}')

                    R_high, I_high = self.model(L_high)
                    if idx % self.print_frequency == 0:
                        hook_number = -1
                    loss = self.loss_fn(R_low, R_high, I_low, I_high, L_low, L_high, hook=hook_number)
                    hook_number = -1  # NOTE: why are we setting hook_number to -1 in both cases?

                    if idx % 8 == 0:
                        print(f"Epoch: {iter} | Step: {idx} | Loss: {loss.item()}")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    idx += 1

                if iter % self.print_frequency == 0:
                    self.test(iter, plot_dir='./images/samples-decom')

                if iter % self.save_frequency == 0:
                    torch.save(self.model.state_dict(), './weights/decom_net.pth')
                    log(f'Checkpoint {iter} saved for decom net!')
                scheduler.step()
                iter_end_time = time.time()
                log(f'Time taken: {iter_end_time - iter_start_time} seconds\t lr: {scheduler.get_last_lr()}')
        except KeyboardInterrupt:
            torch.save(self.model.state_dict(), f'./weights/decom_{iter}_inter.pth')
            log('Saved model before quit')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

    # NOTE: no_grad is from local utils module
    @no_grad
    def test(self, epoch=-1, plot_dir='./images/samples-decom'):
        self.model.eval()
        hook = 0
        for L_low_tensor, L_high_tensor, name in tqdm(self.dataloader_test):
            L_low = L_low_tensor.to(self.device)
            L_high = L_high_tensor.to(self.device)
            R_low, I_low = self.model(L_low)
            R_high, I_high = self.model(L_high)

            if epoch % (self.print_frequency * 10) == 0:
                loss = self.loss_fn(R_low, R_high, I_low, I_high, L_low, L_high, hook)
                hook += 1
                loss = 0
                # WARN: Why are we setting loss to 0 here?

            R_low_np = R_low.detach().cpu().numpy()[0]
            R_high_np = R_high.detach().cpu().numpy()[0]
            I_low_np = I_low.detach().cpu().numpy()[0]
            I_high_np = I_high.detach().cpu().numpy()[0]
            L_low_np = L_low_tensor.numpy()[0]
            L_high_np = L_high_tensor.numpy()[0]
            sample_imgs = np.concatenate((R_low_np, I_low_np, L_low_np,
                                          R_high_np, I_high_np, L_high_np), axis=0)
            filepath = os.path.join(plot_dir, f'{name[0]}_epoch_{epoch}.png')
            split_point = [0, 3, 4, 7, 10, 11, 14]
            img_dim = I_low_np.shape[1:]
            sample(sample_imgs, split=split_point, figure_size=(2, 3),
                   img_dim=img_dim, path=filepath, num=epoch)


if __name__ == '__main__':
    criterion = DecomLoss()
    model = DecomNet()

    parser = BaseParser()
    args = parser.parse()
    # args.checkpoint = None
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    weight_dir = config['weights_dir']
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    if config['checkpoint'] is True:
        pretrain = torch.load(config['decom_net_path'])  # WARN: modified path
        model.load_state_dict(pretrain)
        print('Model loaded from decom_net.pth')

    # WARN: Different from the original code
    root_path_train = config['root_path_train']
    root_path_test = config['root_path_test']
    list_path_train = build_LOLDataset_list_txt(root_path_train)
    list_path_test = build_LOLDataset_list_txt(root_path_test)
    log("Building LOLDataset")

    # WARN: crop size is set to length? and also to_RAM should come from config and not be hardcoded
    dst_train = LOLDataset(root_path_train, list_path_train,
                           crop_size=config['length'], to_RAM=True)

    dst_test = LOLDataset(root_path_test, list_path_test,
                          crop_size=config['length'], to_RAM=True, training=False)

    train_loader = DataLoader(dst_train, batch_size=config['batch_size'],
                              shuffle=True)
    test_loader = DataLoader(dst_test, batch_size=1)

    # WARN: Code skipped here

    trainer = DecomTrainer(config, train_loader, criterion, model, dataloader_test=test_loader)
    if args.mode == 'train':
        # print('Successfully reached training call')
        trainer.train()
    else:
        # print('Successfully reached testing call')
        trainer.test()

#MATCHED