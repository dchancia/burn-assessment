# Burn Assessment Project
# Created by: Daniela Chanci
# Based on: Based on https://github.com/milesial/Pytorch-UNet

import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as f
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from PIL import Image
import json


class BurnDataset(Dataset):
    """
    Class to create our customized dataset
    """

    def __init__(self, inputs_dir, masks_dir, num_classes, train=True):
        self.inputs_dir = inputs_dir
        self.masks_dir = masks_dir
        self.data = os.listdir(self.inputs_dir)
        self.train = train
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_name = self.data[index].split(".")[0]
        input_file = os.path.join(self.inputs_dir, file_name + ".png")
        mask_file = os.path.join(self.masks_dir, file_name + ".png")
        image = Image.open(input_file)  # RGB
        mask = Image.open(mask_file)
        image = np.array(image)
        image = image.transpose(2, 0, 1) / 255
        mask = np.array(mask)
        target = torch.zeros(256, 256)
        target[mask == 0] = 0
        target[mask == 127] = 1
        target[mask == 255] = 2
        im, ground_t = torch.from_numpy(image).type(torch.FloatTensor), target

        return im, ground_t


# Define U-net Blocks
class DownConv(nn.Module):
    """
    One Max Pooling
    Two Convolution -> Batch Normalization -> ReLu
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downblock = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.downblock(x)


class UpConv(nn.Module):

    """"
    One up convolution
    Two Convolution -> Batch Normalization -> ReLu
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = self.in_channels // 2
        self.bilinear = bilinear
        if self.bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.mid_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.ConvTranspose2d(self.in_channels, self.in_channels // 2, kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dif_h = x2.size()[2] - x1.size()[2]
        dif_w = x2.size()[3] - x1.size()[3]
        x1 = f.pad(x1, [dif_w // 2, dif_w - dif_w // 2, dif_h // 2, dif_h - dif_h // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.doubleconv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.doubleconv(x)


class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# Complete model
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.l1 = DoubleConv(self.n_channels, 64)
        self.down1 = DownConv(64, 128)
        self.down2 = DownConv(128, 256)
        self.down3 = DownConv(256, 512)
        self.down4 = DownConv(512, 1024 // factor)
        self.up1 = UpConv(1024, 512 // factor, bilinear)
        self.up2 = UpConv(512, 256 // factor, bilinear)
        self.up3 = UpConv(256, 128 // factor, bilinear)
        self.up4 = UpConv(128, 64, bilinear)
        self.out = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out(x)
        return logits


def train_model(model, device, epochs, lr, dataloader, save_dir):

    start = time.time()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    training_loss = []
    validation_loss = []
    training_accuracy = []
    validation_accuracy = []
    printing_list = []
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):

        print('Epoch {}/{}'.format(epoch+1, epochs))
        printing_list.append('Epoch {}/{}'.format(epoch+1, epochs))
        print('-'*30)
        printing_list.append('-'*30)

        for phase in ['Train', 'Val']:

            running_loss = 0.0
            running_corrects = 0.0

            if phase == 'Train':

                model.train()

                for images, targets in dataloader[phase]:
                    images = images.to(device, dtype=torch.float32)
                    targets = targets.to(device, dtype=torch.long)
                    optimizer.zero_grad()
                    outputs = model(images)
                    _, predictions = torch.max(outputs, 1)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    nn.utils.clip_grad_value_(model.parameters(), 0.1)
                    optimizer.step()
                    running_loss += loss.item() * images.size(0)
                    running_corrects += torch.sum(predictions == targets.data) / (256 ** 2)

            else:

                model.eval()

                for images, targets in dataloader[phase]:
                    images = images.to(device, dtype=torch.float32)
                    targets = targets.to(device, dtype=torch.long)
                    optimizer.zero_grad()

                    with torch.no_grad():
                        outputs = model(images)
                        _, predictions = torch.max(outputs, 1)
                        loss = criterion(outputs, targets)
                    running_loss += loss.item() * images.size(0)
                    running_corrects += torch.sum(predictions == targets.data) / (256 ** 2)

                scheduler.step()

            epoch_loss = running_loss / len(dataloader[phase].dataset)
            epoch_acc = (running_corrects / len(dataloader[phase].dataset)).cpu().numpy()

            print('{} Loss: {:.4f}  Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            printing_list.append('{} Loss: {:.4f}  Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'Train':
                training_loss.append(epoch_loss)
                training_accuracy.append(epoch_acc)
            else:
                validation_loss.append(epoch_loss)
                validation_accuracy.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_weights = copy.deepcopy(model.state_dict())

    train_time = time.time() - start
    print('Training complete in {:.0f}m {:.0f}'.format(train_time // 60, train_time % 60))
    printing_list.append('Training complete in {:.0f}m {:.0f}'.format(train_time // 60, train_time % 60))
    print('Best validation accuracy: {:.4f}'.format(best_acc))
    printing_list.append('Best validation accuracy: {:.4f}'.format(best_acc))

    plt.figure(figsize=(8, 5))
    plt.subplot(1, 2, 1)
    plt.plot(list(range(epochs)), training_loss, color='skyblue', label='Train')
    plt.plot(list(range(epochs)), validation_loss, color='orange', label='Val')
    plt.legend()
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.subplot(1, 2, 2)
    plt.plot(list(range(epochs)), training_accuracy, color='skyblue', label='Train')
    plt.plot(list(range(epochs)), validation_accuracy, color='orange', label='Val')
    plt.legend()
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    fig_dir = os.path.join(save_dir, "training_plot.png")
    plt.savefig(fig_dir)
    plt.show()

    model.load_state_dict(best_model_weights)

    return model, training_loss, training_accuracy, validation_loss, validation_accuracy, printing_list


if __name__ == "__main__":

    # Paths
    file_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
    data_dir = os.path.join(parent_dir, "Data", "RGB", "Dataset_Augmented")
    labels_dir = os.path.join(parent_dir, "Data", "RGB", "Masks_Greyscale_Augmented")
    save_dir = os.path.join(parent_dir, "Segmentation_Results")

    # Model inputs
    batch_size = 4
    device = torch.device("cuda:0")
    learning_rate = 0.001
    n_epochs = 2
    n_classes = 3
    n_channels = 3

    # Create training and validation datasets
    training_dataset = BurnDataset(os.path.join(data_dir, "Train"), os.path.join(labels_dir, "Train"), n_classes, train=True)
    val_dataset = BurnDataset(os.path.join(data_dir, "Val"), os.path.join(labels_dir, "Val"), n_classes, train=False)

    # Create training and validation dataloaders
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    dataloader = {'Train': training_dataloader, 'Val': val_dataloader}

    # Initialize model
    model = UNet(n_channels, n_classes)

    # Training and validation
    segmentation_model, training_loss, training_accuracy, validation_loss, validation_accuracy, printing_list = train_model(model,
                                                device, n_epochs, learning_rate, dataloader, save_dir=save_dir)

    # Save model
    torch.save(segmentation_model.state_dict(), os.path.join(save_dir, "UNet_std.pth"))
    summary_model = {'Training Loss': list(map(str, training_loss)), 'Training Accuracy': list(map(str, training_accuracy)),
                     'Validation Loss': list(map(str, validation_loss)), 'Validation Accuracy': list(map(str, validation_accuracy))}
    json = json.dumps(summary_model)
    file1 = open(os.path.join(save_dir, "summary.txt"), "w")
    file1.write(str(summary_model))
    file1.close()
    file2 = open(os.path.join(save_dir, "summary.json"), "w")
    file2.write(json)
    file2.close()

    file3 = open(os.path.join(save_dir, "print.txt"), "w")
    for lines in printing_list:
        file3.write(lines)
        file3.write('\n')
    file3.close()

