"""
# pyton 3
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
# import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from skimage import io
# from torchvision.models import mobilenet_v2
from custom_mobilenet import CustomMobileNet

from PIL import Image
import numpy as np

import pandas as pd

# import matplotlib.pyplot as plt
from tqdm import tqdm


class DriveDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
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
        # self.data = self.data['filenames'].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data.iloc[idx]
        # RGB image
        folder, file = row['filenames'].split('/')
        img_rgb_path = os.path.join(
            self.root_dir, 'Images', folder, 'rgb', file)
        img_rgb = Image.open(img_rgb_path)

        if self.transform:
            img_rgb = self.transform(img_rgb)

        # _, _, throttle, steering = img_name.split('/')[-1].split('_')
        throttle = row['throttle']
        steering = row['steer']
        action_left = row['action_left']
        action_right = row['action_right']
        action_forward = row['action_forward']
        no_action = row['no_action']

        sample = (img_rgb,
                  torch.tensor([
                      float(action_left), float(action_right),
                      float(action_forward), float(no_action)
                  ]),
                  torch.tensor([
                      float(throttle), float(steering)
                  ]))

        return sample


if __name__ == '__main__':
    # model = mobilenet_v2()
    model = CustomMobileNet(pretrained=True)

    # epoch, arch, state_dict = torch.load(
    #     'weights/mob_drive_9.pth.tar').values()
    # model.load_state_dict(state_dict)

    model.cuda()

    train_dir = '/home/nubol23/Documents/DriveDatasetStable/Train/train_dataset_final.csv'
    val_dir = '/home/nubol23/Documents/DriveDatasetStable/Val/val_dataset_final.csv'

    train_loader = torch.utils.data.DataLoader(
        dataset=DriveDataset(train_dir, transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            lambda T: T[:3]
        ])),
        batch_size=64,
        shuffle=True,
        num_workers=12,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=DriveDataset(val_dir, transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])),
        batch_size=64,
        shuffle=False,
        num_workers=12,
        pin_memory=True
    )

    criterion = nn.MSELoss().cuda()

    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=1e-3,
    #     momentum=0.9
    # )
    optimizer = torch.optim.Adam(model.parameters())

    losses = []

    for epoch in range(50):
        # for epoch in range(10):
        start = time.time()

        model.train()
        train_loss = 0
        train_progress = tqdm(enumerate(train_loader),
                              desc="train",
                              total=len(train_loader))
        for i, (X, actions, y) in train_progress:
            X = X.cuda(non_blocking=True)
            actions = actions.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            y_hat = model(X, actions)

            loss1 = criterion(y_hat[:, 0], y[:, 0])
            loss2 = criterion(y_hat[:, 1], y[:, 1])
            loss = (loss1 + loss2)/2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += float(loss.detach())
            train_progress.set_postfix(loss=(train_loss/(i+1)))

        model.eval()

        val_loss = 0
        with torch.no_grad():
            model.eval()
            val_progress = tqdm(enumerate(val_loader),
                                desc="val",
                                total=len(val_loader))
            for i, (X, actions, y) in val_progress:
                X = X.cuda(non_blocking=True)
                actions = actions.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                y_hat = model(X, actions)

                loss1 = criterion(y_hat[:, 0], y[:, 0])
                loss2 = criterion(y_hat[:, 1], y[:, 1])
                loss = (loss1 + loss2) / 2

                val_loss += float(loss)
                val_progress.set_postfix(loss=(val_loss/(i+1)))

        end = time.time()

        t_loss = train_loss / len(train_loader)
        v_loss = val_loss / len(val_loader)
        print('epoch:', epoch, 'L:', t_loss, v_loss, 'Time:', end-start)

        torch.save(
            {
                'epoch': epoch,
                'arch': 'mobilenet_custom',
                'state_dict': model.state_dict()
            },
            f'weights_extra/mob_drive_{epoch}.pth.tar')
            # f'weights/mob_drive_{epoch}.pth.tar')
        losses.append([epoch, t_loss, v_loss])
        np.save('hist_drive', np.array(losses))
