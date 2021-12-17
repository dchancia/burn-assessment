import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import numpy as np
import cv2
from skimage.feature import greycomatrix, greycoprops
from ResNet18_Textures import ModifiedResNet
import sklearn.metrics


# Paths for dataset
file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
data_dir = data_dir = os.path.join(parent_dir, "Data", "HUSD", "Test")
weights_path = os.path.join(parent_dir, "Classification_Results", "resnet_text_std.pth")


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


if __name__ == "__main__":

    resnet_input_size = 224
    batch_size = 8
    num_classes = 3
    device = torch.device("cuda:0")
    acc = 0.0
    recall = 0.0
    precision = 0.0
    f1 = 0.0
    confusion = np.zeros((3, 3))
    preds_total = ""
    num_tfeatures = 30
    feature_extract = True

    # Normalization
    preprocess = transforms.Compose([
            transforms.Resize(resnet_input_size),
            transforms.CenterCrop(resnet_input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Create testing datasets
    test_burn_dataset = ImageFolderWithFeatures(data_dir, transform=preprocess)

    # Create training and validation dataloaders
    test_dataloader = DataLoader(test_burn_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = ModifiedResNet(num_classes, num_tfeatures)

    model.load_state_dict(torch.load(weights_path))
    model = model.to(device)
    model.eval()

    # Testing model

    for inputs, labels, features in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        features = features.to(device)
        outputs = model(inputs, features)
        _, predictions = torch.max(outputs, 1)
        acc += torch.sum(predictions == labels.data)
        preds = predictions.data.cpu().numpy()
        targets = labels.data.cpu().numpy()
        recall += sklearn.metrics.recall_score(targets.flatten(), preds.flatten(),
                                               average="macro", zero_division=0)
        precision += sklearn.metrics.precision_score(targets.flatten(), preds.flatten(),
                                                     average="macro", zero_division=0)
        f1 += sklearn.metrics.f1_score(targets.flatten(), preds.flatten(),
                                       average="macro", zero_division=0)
        for j in range(len(inputs)):
            confusion[preds[j], targets[j]] += 1
    print(confusion)

    print('Testing Accuracy: {:.4f}'.format(acc / len(test_dataloader.dataset)))
    print('Recall: {:.4f}'.format(recall / len(test_dataloader)))
    print('Precision: {:.4f}'.format(precision / len(test_dataloader)))
    print('f1: {:.4f}'.format(f1 / len(test_dataloader)))
