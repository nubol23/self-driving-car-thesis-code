from abc import ABC

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn.functional as F
from semseg_model import CustomMobilenetSemseg

from PIL import Image, ImageOps
import os
import numpy as np
import pandas as pd
import time

from tqdm import tqdm


class DriveSemsegDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None, stills=True, train=True):
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
            self.data = self.data['filenames'].tolist()
        else:
            if train:
                self.data = self.data[168000:]  # Extraemos los que tienen autos
                # self.data = self.data.sample( int(round(len(self.data)*(2/3))) )
            else:
                self.data = self.data[:40000]
            self.data = self.data.query('throttle != 0.0')
            # self.data = self.data.tail(1000)
            self.data = self.data['filenames'].tolist()
            
        #print(self.data[:15])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        h_flip = np.random.random() < 0.5

        # RGB image
        folder, file = self.data[idx].split('/')
        img_rgb_path = os.path.join(self.root_dir, 'Images', folder, 'rgb', file)
        img_rgb = Image.open(img_rgb_path)
        if h_flip:
            img_rgb = ImageOps.mirror(img_rgb)
        if self.transform:
            img_rgb = self.transform(img_rgb)

        # MASK
        img_semseg_path = os.path.join(self.root_dir, 'Images', folder, 'mask', file)
        img_semseg = Image.open(img_semseg_path)
        if h_flip:
            img_semseg = ImageOps.mirror(img_semseg)
        img_semseg = np.asarray(img_semseg)[:, :, 0].copy()
        sample = (img_rgb, torch.from_numpy(img_semseg))

        return sample


if __name__ == '__main__':
    np.random.seed(42)
    
    model = CustomMobilenetSemseg((180, 240), pretrained=False)
    model = model.cuda()

    train_dir = '/home/nubol23/Documents/DriveDatasetStable/Train/train_dataset.csv'
    val_dir = '/home/nubol23/Documents/DriveDatasetStable/Val/val_dataset.csv'

    train_loader = torch.utils.data.DataLoader(
        dataset=DriveSemsegDataset(train_dir, transforms.Compose([
            transforms.ToTensor(),
            lambda T: T[:3],
        ]), stills=False),
        batch_size=64,
        shuffle=True,
        num_workers=12,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=DriveSemsegDataset(val_dir, transforms.Compose([
            transforms.ToTensor(),
        ]), stills=False, train=False),
        batch_size=64,
        shuffle=True,
        num_workers=12,
        pin_memory=True
    )

    criterion = nn.CrossEntropyLoss().cuda()  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, amsgrad=True)

    losses = []
    
    train_len = len(train_loader)
    val_len = len(val_loader)
    for epoch in range(50):
        start = time.time()

        model.train()
        train_loss = 0
        
        train_progress = tqdm(enumerate(train_loader), desc="train", total=train_len)
        for i, (X, y) in train_progress:
            # torch.Size([batch_size, 3, 180, 240])
            # torch.Size([batch_size, 1, 180, 240])
            X = X.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            y_hat = model(X)

            loss = criterion(y_hat, y.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_val = loss.detach()
            train_loss += float(loss_val)
            
            train_progress.set_postfix(loss=(train_loss/(i+1))) # Loss info
        
        torch.save(
            {
                'epoch': epoch,
                'arch': 'mobilenet_depth',
                'state_dict': model.state_dict()
            },
            f'weights/s_mob_{epoch}.pth.tar')
        
        val_loss = 0
        with torch.no_grad():
            model.eval()
            val_progress = tqdm(enumerate(val_loader), desc="val", total=val_len)
            for i, (X, y) in val_progress:
                X = X.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                y_hat = model(X)

                loss = criterion(y_hat, y.long())

                val_loss += float(loss)
                
                val_progress.set_postfix(loss=(val_loss/(i+1))) # Loss info

        end = time.time()

        t_loss = train_loss / len(train_loader)
        v_loss = val_loss / len(val_loader)
        print('epoch:', epoch, 'L:', t_loss, v_loss, 'Time:', end - start)

        losses.append([epoch, t_loss, v_loss])
        np.save('hist', np.array(losses))
