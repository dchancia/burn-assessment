import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import json
import cv2
from skimage.feature import greycomatrix, greycoprops


class ImageFolderWithFeatures(datasets.ImageFolder):

    # From: https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithFeatures, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        texture_features = extract_text_features(path)
        # make a new tuple that includes original and the path
        tuple_with_features = (original_tuple + (texture_features,))
        return tuple_with_features


class ModifiedResNet(nn.Module):
    def __init__(self, num_classes, text_features):
        super().__init__()
        self.num_classes = num_classes
        self.text_features = text_features
        self.num_features = 512
        self.new_model = torch.nn.Sequential(*(list(models.resnet18(pretrained=True).children())[:-1]))
        self.fc = nn.Linear(in_features=512+self.text_features, out_features=self.num_classes)

    def forward(self, image, texture):
        x1 = self.new_model(image)
        x1 = torch.flatten(x1, 1)
        x2 = texture
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x


def extract_text_features(im_path):
    image = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2GRAY)
    im_array = np.array(image, dtype=np.uint8)[:300,:]

    dist = [1, 2]
    angles = [0, np.pi / 4, np.pi / 2]

    g = greycomatrix(im_array, dist, angles, 256, symmetric=True, normed=True)

    contrast = greycoprops(g, "contrast")
    homogeneity = greycoprops(g, "homogeneity")
    ASM = greycoprops(g, "ASM")
    energy = greycoprops(g, "energy")
    dissimilarity = greycoprops(g, "dissimilarity")

    text_features = []

    for i in range(len(dist)):
        for j in range(len(angles)):
            text_features.append(contrast[i][j])
            text_features.append(homogeneity[i][j])
            text_features.append(ASM[i][j])
            text_features.append(energy[i][j])
            text_features.append(dissimilarity[i][j])

    feat_array = np.expand_dims(np.array((text_features)), 1)
    features = torch.from_numpy(feat_array).type(torch.FloatTensor)
    features = torch.squeeze(features, 1)
    return features


def initialize_resnet(num_classes, num_text, feature_ext):
    'Initialize ResNet model'
    model = ModifiedResNet(num_classes, num_text)
    if feature_ext:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Linear(in_features=512+num_text, out_features=num_classes)
    input_size = 224
    return model, input_size


def create_optimizer(model, feature_ext, lr):
    'Create optimizer for model'
    params_to_update = model.parameters()
    if feature_ext:
        params_to_update = []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad == True:
                params_to_update.append(parameter)
    optimizer = optim.Adam(params_to_update, lr=lr)
    return optimizer


def train_model(model, dataloader, optimizer, num_epochs, device, save_dir):

    start = time.time()
    criterion = nn.CrossEntropyLoss()

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    printing_list = []
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Iterate over epoch
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 30)
        printing_list.append('Epoch {}/{}'.format(epoch+1, num_epochs))
        printing_list.append('-' * 30)

        for phase in ['Train', 'Val']:

            if phase == 'Train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data
            for data in dataloader[phase]:
                inputs, labels, texture = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                texture = texture.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs, texture)
                    loss = criterion(outputs, labels)
                    _, predictions = torch.max(outputs, 1)

                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == labels.data)

            epoch_loss = running_loss / len(dataloader[phase].dataset)
            epoch_acc = (running_corrects / len(dataloader[phase].dataset)).cpu().numpy()

            print('{} Loss: {:.4f}  Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            printing_list.append('{} Loss: {:.4f}  Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'Train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                # scheduler.step()
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_weights = copy.deepcopy(model.state_dict())

        print()

    train_time = time.time() - start
    print('Training complete in {:.0f}m {:.0f}'.format(train_time // 60, train_time % 60))
    print('Best validation accuracy: {:.4f}'.format(best_acc))
    printing_list.append('Training complete in {:.0f}m {:.0f}'.format(train_time // 60, train_time % 60))
    printing_list.append('Best validation accuracy: {:.4f}'.format(best_acc))

    plt.subplot(1, 2, 1)
    plt.plot(list(range(num_epochs)), train_loss, color='skyblue', label='Train')
    plt.plot(list(range(num_epochs)), val_loss, color='orange', label='Val')
    plt.legend()
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.subplot(1, 2, 2)
    plt.plot(list(range(num_epochs)), train_acc, color='skyblue', label='Train')
    plt.plot(list(range(num_epochs)), val_acc, color='orange', label='Val')
    plt.legend()
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    fig_dir = os.path.join(save_dir, "training_plot.png")
    plt.savefig(fig_dir)
    plt.show()

    model.load_state_dict(best_model_weights)

    return model, train_loss, train_acc, val_loss, val_acc, printing_list


if __name__ == "__main__":

    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    # Paths
    file_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
    data_dir = os.path.join(parent_dir, "Data", "HUSD")
    save_dir = os.path.join(parent_dir, "Classification_Results")

    num_classes = 3
    batch_size = 8
    num_epochs = 2
    num_tfeatures = 30
    feature_extract = True
    learning_rate = 0.00001
    momentum = 0.9

    # Initialize model
    pretrained_model, resnet_input_size = initialize_resnet(num_classes, num_tfeatures, feature_extract)
    device = torch.device("cuda:0")
    pretrained_model = pretrained_model.to(device)
    # print(pretrained_model)

    # Load Data
    preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Create training and validation datasets
    train_dataset = ImageFolderWithFeatures(os.path.join(data_dir, 'Train'), transform=preprocess)
    val_dataset = ImageFolderWithFeatures(os.path.join(data_dir, 'Val'), transform=preprocess)

    # Create training and validation dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    dataloader = {'Train': train_dataloader, 'Val': val_dataloader}

    # Create optimizer
    optimizer = create_optimizer(pretrained_model, feature_extract, learning_rate)


    # Training and Validation
    ft_model, train_loss, train_acc, val_loss, val_acc, print_list = train_model(pretrained_model, dataloader,
                                                                                 optimizer, num_epochs, device, save_dir)

    print_list.append('\n')
    print_list.append("PyTorch Version: {}".format(torch.__version__))
    print_list.append("Torchvision Version: {}".format(torchvision.__version__))
    print_list.append("ResNet_101 with texture features")
    print_list.append('Epochs: {:.0f}, Batch Size: {:.0f}, LR: {:.8f}, Adam'.format(num_epochs, batch_size, learning_rate))

    # Save model
    torch.save(ft_model.state_dict(), os.path.join(save_dir, "resnet_text_std.pth"))
    torch.save(ft_model, os.path.join(save_dir, "resnet_text.pth"))

    summary_model = {'Training Loss': list(map(str, train_loss)),
                     'Training Accuracy': list(map(str, train_acc)),
                     'Validation Loss': list(map(str, val_loss)),
                     'Validation Accuracy': list(map(str, val_acc))}
    json = json.dumps(summary_model)
    file1 = open(os.path.join(save_dir, "summary.txt"), "w")
    file1.write(str(summary_model))
    file1.close()
    file2 = open(os.path.join(save_dir, "summary.json"), "w")
    file2.write(json)
    file2.close()

    file3 = open(os.path.join(save_dir, "print.txt"), "w")
    for lines in print_list:
        file3.write(lines)
        file3.write('\n')
    file3.close()


