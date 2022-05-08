from __future__ import print_function
from __future__ import division
from cgi import test

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image


print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# Top level data directory.
data_dir = "./data/hymenoptera_data"

# Models from [resnet18, resnet34]
model_name = "resnet18"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Parameters
num_classes = 2
batch_size = 8
num_epochs = 15

# Flag for feature extracting. 
feature_extract = True


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18 """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif model_name == "resnet34":
        """ Resnet34 """
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    return model_ft, input_size



id_to_specie = {}
with open('data/oxford-iiit-pet/annotations/list.txt') as f:
    lines = f.readlines()
    for line_idx, line in enumerate(lines[6:],6):
        tokens = line.split(' ')
        id_to_specie[tokens[1]] = tokens[2]

print(id_to_specie)


def main():
    # Load pretrained dataset
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    
    trainingval_data = torchvision.datasets.OxfordIIITPet(
        root = "data",
        split = 'trainval',
        download = True,
        transform = transforms.Compose([
            

            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    test_data = torchvision.datasets.OxfordIIITPet(
        root = "data",
        split = 'test',
        download = True,
        transform = transforms.Compose([
            transforms.Resize(input_size),
            #transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    # Load training and validation datasets
    train_dataset, val_dataset = torch.utils.data.random_split(trainingval_data, [len(trainingval_data)//2,len(trainingval_data)//2  ])
    
    



    # Create training and validation dataloaders
    dataloaders_dict = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
                        'val':  torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)}
    
    
    print(dataloaders_dict['train'])
    train_features, train_labels = next(iter(dataloaders_dict["train"]))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    #img2 = Image.fromarray(img, 'RGB')
    #img2.show()

    plt.imshow(img[0], cmap="Reds")
    plt.show()
    plt.imshow(img[1], cmap="Greens")
    plt.show()
    plt.imshow(img[2], cmap="Blues")
    plt.show()


    print(f"Label: {label}")
        
    return 0


if __name__ == '__main__':
    main()