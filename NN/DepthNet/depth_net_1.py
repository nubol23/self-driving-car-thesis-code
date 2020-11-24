import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from skimage import io
from models import CustomMobilenet

from PIL import Image, ImageOps
import os
import numpy as np
import pandas as pd
import time

from tqdm import tqdm

import matplotlib.pyplot as plt


class DriveDepthDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None, stills=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        temp = root_dir.split('/')
        self.root_dir = '/'.join(temp[:-1])
        self.transform = transform

        self.data = pd.read_csv(root_dir)
        if stills:
            self.data = self.data['n_id'].tolist()
        else:
            self.data = self.data.query('throttle != 0.0')['n_id'].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        h_flip = np.random.random() < 0.5

        # RGB image
        img_rgb_path = os.path.join(self.root_dir, 'rgb', f'{self.data[idx]}.png')
        img_rgb = Image.open(img_rgb_path)
        if h_flip:
            img_rgb = ImageOps.mirror(img_rgb)
        if self.transform:
            img_rgb = self.transform(img_rgb)

        # DEPTH
        img_depth_path = os.path.join(self.root_dir, 'depth', f'{self.data[idx]}.png')
        img_depth = Image.open(img_depth_path)
        if h_flip:
            img_depth = ImageOps.mirror(img_depth)
        img_depth = np.transpose(np.asarray(img_depth, dtype=np.float32), (2, 0, 1))
        target = img_depth[0, :, :] + img_depth[1, :, :] * 256 + img_depth[2, :, :] * 256 * 256
        target = np.clip((target / (256 * 256 * 256 - 1)) * 1000, None, 30)
        target = torch.from_numpy(target).float()

        # sample = (img_rgb[:3], target)
        sample = (img_rgb, target.view(1, target.shape[0], target.shape[1]))

        return sample

import matplotlib.pyplot as plt

if __name__ == '__main__':

    # trans = transforms.Compose([
    #     transforms.ToTensor(),
    # ])

    # TESTS
    # dat = DriveDepthDataset('/home/nubol23/Documents/DriveDatasetWhole/raw/train/train_data.csv',
    #                         trans)
    # x, y = dat[0]
    # print(x.shape)
    # plt.imshow(x.permute(1, 2, 0))
    # plt.show()
    # print(y.shape)
    # plt.imshow(np.log(y), cmap='gray')
    # plt.show()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    model = CustomMobilenet((180, 240), pretrained=False)

    # model.eval()
    # out = model(x.reshape(1, 3, 180, 240))
    # print(out.shape)
    # plt.imshow(out[0,0], cmap='gray')
    # plt.show()

    model.cuda()

    train_dir = '/home/nubol23/Documents/DriveDatasetWhole/raw/train/train_data.csv'
    val_dir = '/home/nubol23/Documents/DriveDatasetWhole/raw/val/val_data.csv'

    train_loader = torch.utils.data.DataLoader(
        dataset=DriveDepthDataset(train_dir, transforms.Compose([
            transforms.ToTensor(),
            lambda T: T[:3],
            normalize
        ])),
        batch_size=64,
        shuffle=True,
        num_workers=12,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=DriveDepthDataset(val_dir, transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])),
        batch_size=64,
        shuffle=True,
        num_workers=12,
        pin_memory=True
    )

    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters())

    losses = []

    for epoch in range(50):
        start = time.time()

        model.train()
        train_loss = 0
        for i, (X, y) in tqdm(enumerate(train_loader)):
            # torch.Size([16, 3, 180, 240])
            # torch.Size([16, 1, 180, 240])
            X = X.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            y_hat = model(X)

            loss = criterion(y_hat, y)
            train_loss += float(loss.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss = 0
        with torch.no_grad():
            model.eval()
            for i, (X, y) in enumerate(val_loader):
                X = X.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                y_hat = model(X)

                loss = criterion(y_hat, y)
                val_loss += float(loss)

        end = time.time()

        t_loss = train_loss / len(train_loader)
        v_loss = val_loss / len(val_loader)
        print('epoch:', epoch, 'L:', t_loss, v_loss, 'Time:', end - start)

        torch.save(
            {
                'epoch': epoch,
                'arch': 'mobilenet_depth',
                'state_dict': model.state_dict()
            },
            f'weights/c_mob_{epoch}.pth.tar')
        losses.append([epoch, t_loss, v_loss])
        np.save('hist', np.array(losses))