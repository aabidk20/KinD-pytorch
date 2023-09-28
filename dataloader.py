import os
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import cv2
import shutil
import time
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from utils import *


# WARN: better handle splitting of paths in classes

class CustomDataset(Dataset):
    def __init__(self, datapath):
        super().__init__()
        self.datapath = datapath
        self.img_path = [os.path.join(datapath, f) for f in os.listdir(datapath) if
                         any(filetype in f.lower() for filetype in ['jpg', 'png', 'jpeg', 'bmp'])]
        # better extract name
        self.name = [os.path.splitext(f)[0] for f in os.listdir(datapath) if
                     any(filetype in f.lower() for filetype in ['jpg', 'png', 'jpeg', 'bmp'])]
        print(self.img_path)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        datafiles = self.img_path[idx]

        # Open the image and convert it to RGB(image might not be RGB)
        img = Image.open(datafiles).convert('RGB')

        # Convert the image to numpy array, take transpose to make
        # it channels first and normalize
        img = np.asarray(img, np.float32).transpose((2, 0, 1)) / 255.
        return img, self.name[idx]


class LOLDataset(Dataset):
    def __init__(self, root, list_path, crop_size=256, to_RAM=False, training=True):
        super(LOLDataset, self).__init__()
        self.training = training
        self.to_RAM = to_RAM
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size

        with open(list_path, 'r') as f:
            # self.pairs = f.readlines()[0],
            self.pairs = f.readlines()
        self.files = []
        for pair in self.pairs:
            lr_path, hr_path = pair.split(",")
            hr_path = hr_path[:-1]
            name = lr_path.split("\\")[-1][:-4]
            lr_file = os.path.join(self.root, lr_path)
            hr_file = os.path.join(self.root, hr_path)
            self.files.append({
                "lr": lr_file,
                "hr": hr_file,
                "name": name
            })
            self.data = []
            if self.to_RAM:
                for i, fileinfo in enumerate(self.files):
                    name = fileinfo["name"]
                    lr_img = Image.open(fileinfo["lr"])
                    hr_img = Image.open(fileinfo["hr"])
                    self.data.append({
                        "lr": lr_img,
                        "hr": hr_img,
                        "name": name
                    })
        log(f"Finished loading {len(self.files)} images to RAM")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        datafiles = self.files[idx]

        if not self.to_RAM:  # data is not in RAM
            name = datafiles["name"]
            lr_img = Image.open(
                datafiles["lr"])  # NOTE: custom dataset might not be RGB, so we converted it. LOL is RGB
            hr_img = Image.open(datafiles["hr"])
        else:  # data is in RAM
            name = self.data[idx]["name"]
            lr_img = self.data[idx]["lr"]
            hr_img = self.data[idx]["hr"]

        if self.crop_size > 0:
            h_offset = random.randint(0, lr_img.size[1] - self.crop_size)
            w_offset = random.randint(0, lr_img.size[0] - self.crop_size)

            crop_box = (w_offset, h_offset, w_offset + self.crop_size, h_offset + self.crop_size)
            lr_crop = lr_img
            hr_crop = hr_img
            if self.training is True:
                lr_crop = lr_img.crop(crop_box)
                hr_crop = hr_img.crop(crop_box)
                rand_mode = np.random.randint(0, 8)
                lr_crop = data_augmentation(lr_crop, rand_mode)
                hr_crop = data_augmentation(hr_crop, rand_mode)

        lr_crop = np.asarray(lr_crop, np.float32).transpose((2, 0, 1)) / 255.
        hr_crop = np.asarray(hr_crop, np.float32).transpose((2, 0, 1)) / 255.
        return lr_crop, hr_crop, name


class LOLDataset_Decom(Dataset):
    def __init__(self, root, list_path, crop_size=256,
                 to_RAM=False, training=True):
        super().__init__()
        self.training = training
        self.to_RAM = to_RAM
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        with open(list_path, 'r') as f:
            self.pairs = f.readlines()
        self.files = []
        for pair in self.pairs:
            lr_path_R, lr_path_I, hr_path_R, hr_path_I = pair.split(",")
            hr_path_I = hr_path_I[:-1]
            name = lr_path_R.split("\\")[-1][:-4]
            lr_file_R = os.path.join(self.root, lr_path_R)
            lr_file_I = os.path.join(self.root, lr_path_I)
            hr_file_R = os.path.join(self.root, hr_path_R)
            hr_file_I = os.path.join(self.root, hr_path_I)
            self.files.append({
                "lr_R": lr_file_R,
                "lr_I": lr_file_I,
                "hr_R": hr_file_R,
                "hr_I": hr_file_I,
                "name": name
            })
        self.data = []
        if self.to_RAM:
            for i, fileinfo in enumerate(self.files):
                name = fileinfo["name"]
                lr_img_R = Image.open(fileinfo["lr_R"])
                hr_img_R = Image.open(fileinfo["hr_R"])
                lr_img_I = Image.open(fileinfo["lr_I"]).convert('L')
                hr_img_I = Image.open(fileinfo["hr_I"]).convert('L')
                self.data.append({
                    "lr_R": lr_img_R,
                    "lr_I": lr_img_I,
                    "hr_R": hr_img_R,
                    "hr_I": hr_img_I,
                    "name": name
                })
            log("Finished loading all images to RAM")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        datafiles = self.files[idx]

        if not self.to_RAM:  # data is not in RAM
            name = datafiles["name"]
            lr_img_R = Image.open(datafiles["lr_R"])
            hr_img_R = Image.open(datafiles["hr_R"])
            lr_img_I = Image.open(datafiles["lr_I"]).convert('L')
            hr_img_I = Image.open(datafiles["hr_I"]).convert('L')
        else:  # data is in RAM
            name = self.data[idx]["name"]
            lr_img_R = self.data[idx]["lr_R"]
            lr_img_I = self.data[idx]["lr_I"]
            hr_img_R = self.data[idx]["hr_R"]
            hr_img_I = self.data[idx]["hr_I"]

        if self.crop_size > 0:
            h_offset = random.randint(0, lr_img_R.size[1] - self.crop_size)
            w_offset = random.randint(0, lr_img_R.size[0] - self.crop_size)

            crop_box = (w_offset, h_offset, w_offset + self.crop_size, h_offset + self.crop_size)
            lr_crop_R = lr_img_R
            lr_crop_I = lr_img_I
            hr_crop_R = hr_img_R
            hr_crop_I = hr_img_I
            if self.training is True:
                lr_crop_R = lr_crop_R.crop(crop_box)
                lr_crop_I = lr_crop_I.crop(crop_box)
                hr_crop_R = hr_crop_R.crop(crop_box)
                hr_crop_I = hr_crop_I.crop(crop_box)
                rand_mode = np.random.randint(0, 8)
                lr_crop_R = data_augmentation(lr_crop_R, rand_mode)
                lr_crop_I = data_augmentation(lr_crop_I, rand_mode)
                hr_crop_R = data_augmentation(hr_crop_R, rand_mode)
                hr_crop_I = data_augmentation(hr_crop_I, rand_mode)

        lr_crop_R = np.asarray(lr_crop_R, np.float32).transpose((2, 0, 1)) / 255.
        lr_crop_I = np.expand_dims(np.asarray(lr_crop_I, np.float32), axis=0) / 255.
        hr_crop_R = np.asarray(hr_crop_R, np.float32).transpose((2, 0, 1)) / 255.
        hr_crop_I = np.expand_dims(np.asarray(hr_crop_I, np.float32), axis=0) / 255.
        return lr_crop_R, lr_crop_I, hr_crop_R, hr_crop_I, name


def build_LOLDataset_list_txt(dst_dir):
    log(f"Building LOLDataset list txt in {dst_dir}")
    lr_dir = os.path.join(dst_dir, 'low')
    hr_dir = os.path.join(dst_dir, 'high')
    img_lr_path = [os.path.join('low', name) for name in os.listdir(lr_dir)]
    img_hr_path = [os.path.join('high', name) for name in os.listdir(hr_dir)]
    list_path = os.path.join(dst_dir, 'pair_list.csv')
    with open(list_path, 'w') as f:
        for lr_path, hr_path in zip(img_lr_path, img_hr_path):
            f.write(f"{lr_path},{hr_path}\n")
    log(f"Finished building LOLDataset list txt in {dst_dir}")
    return list_path


def build_LOLDataset_Decom_list_txt(dst_dir):
    log(f"Building LOLDataset_Decom list txt in {dst_dir}")
    dir_lists = []
    tail = ['low\\R', 'low\\I', 'high\\R', 'high\\I']
    for t in tail:
        dir_lists.append(os.path.join(dst_dir, t))
    img_path = [[], [], [], []]
    for i, directory in enumerate(dir_lists):
        for name in os.listdir(directory):
            path = os.path.join(tail[i], name)
            img_path[i].append(path)
    list_path = os.path.join(dst_dir, 'pair_list.csv')
    with open(list_path, 'w') as f:
        for lr_R, lr_I, hr_R, hr_I in zip(*img_path):
            f.write(f"{lr_R},{lr_I},{hr_R},{hr_I}\n")
    log(f"Finished building LOLDataset_Decom list txt in {dst_dir}.{len(img_path[0])} pairs in total")
    return list_path


def divide_dataset(dst_dir):
    log(f"Dividing dataset in {dst_dir}")
    lr_dir_R = os.path.join(dst_dir, 'low', 'R')
    lr_dir_I = os.path.join(dst_dir, 'low', 'I')
    hr_dir_R = os.path.join(dst_dir, 'high', 'R')
    hr_dir_I = os.path.join(dst_dir, 'high', 'I')

    for name in os.listdir(dst_dir):
        path = os.path.join(dst_dir, name)
        name, ext = os.path.splitext(name)  # DIFF
        item = name.split('_')
        if item[0] == 'high' and item[-1] == 'R':
            shutil.move(path, os.path.join(hr_dir_R, item[1] + ext))
        elif item[0] == 'high' and item[-1] == 'I':
            shutil.move(path, os.path.join(hr_dir_I, item[1] + ext))
        elif item[0] == 'low' and item[-1] == 'R':
            shutil.move(path, os.path.join(lr_dir_R, item[1] + ext))
        elif item[0] == 'low' and item[-1] == 'I':
            shutil.move(path, os.path.join(lr_dir_I, item[1] + ext))
    log(f"Finished dividing dataset in {dst_dir}")


def change_name(dst_dir):
    dir_lists = []
    dir_lists.append(os.path.join(dst_dir, 'low', 'R'))
    dir_lists.append(os.path.join(dst_dir, 'low', 'I'))
    dir_lists.append(os.path.join(dst_dir, 'high', 'R'))
    dir_lists.append(os.path.join(dst_dir, 'high', 'I'))
    for directory in dir_lists:
        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            name, ext = os.path.splitext(name)
            item = name.split('_')
            os.rename(path, os.path.join(directory, item[1] + ext))
    log(f"Finished changing name in {dst_dir}")


if __name__ == '__main__':
    # NOTE: skipped code
    # NOTE: hard-coded paths.change to read from config
    # NOTE: revise this part

    root_path_train = r'/home/aabid/Documents/Datasets/LOLdataset/our485'
    root_path_test = r'/home/aabid/Documents/Datasets/LOLdataset/eval15'
    list_path_train = build_LOLDataset_list_txt(root_path_train)
    list_path_test = build_LOLDataset_list_txt(root_path_test)
    batch_size = 2
    log(f"Building train dataloader with batch size {batch_size}")

    # WARN: training is set to True by default.
    dst_train = LOLDataset(root_path_train, list_path_train, crop_size=128, to_RAM=False)

    # WARN: training should be False here?
    dst_test = LOLDataset(root_path_test, list_path_test, crop_size=128, to_RAM=False, training=False)

    # WARN: should shuffle here
    trainloader = DataLoader(dst_train, batch_size=batch_size)
    testloader = DataLoader(dst_test, batch_size=1)
    plt.ion()
    for i, data in enumerate(trainloader):
        _, imgs, name = data
        img = imgs[0].numpy()
        sample(imgs[0], figure_size=(1, 1), img_dim=128)

# MATCHED
