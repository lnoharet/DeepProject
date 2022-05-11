from __future__ import print_function
from __future__ import division
from cgi import test
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from torchvision.io import read_image
import matplotlib.pyplot as plt
import time
import os
import copy
from glob import glob
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, img_paths, labels, input_size, split):
            
        self.img_labels = labels
        self.img_paths = img_paths

        if split == 'trainval':
            self.transform = transforms.Compose([
                # TODO: kolla om det är rätt transforms
                transforms.RandomResizedCrop(input_size),
                #transforms.RandomHorizontalFlip(),
                #transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else: 
            self.transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = load_image(self.img_paths[idx] + '.jpg')
        label = self.img_labels[idx]    
        image = self.transform(image)
        return image, label



print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# Top level data directory.
data_dir = "./data/oxford-iiit-pet"

# Models from [resnet18, resnet34]
model_name = "resnet18"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# Parameters
num_classes = 2
batch_size = 8
num_epochs = 15

# Flag for feature extracting. 
feature_extract = False

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []
    train_loss_history = []
    val_loss_history = []   

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            count = 0
            for inputs, labels in dataloaders[phase]:
                count+=1
                #if count % 10 == 0:
                #    print(count)
                #    print(len(inputs))
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            
            

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
            else:
                train_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_loss_history, val_loss_history

def test_model(model, dataloaders):
    model.eval()
    acc_history = []
    running_corrects = 0
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

        acc_history.append(running_corrects/len(dataloaders['test'].dataset))
    return acc_history


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

def load_image(filename) :
    img = Image.open(filename)
    img = img.convert('RGB')
    return img

def initialize_training():
    # Load pretrained dataset
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    model_ft = model_ft.to(device)


    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                #print("\t",name)
    #else:
    #    for name,param in model_ft.named_parameters():
    #        if param.requires_grad == True:
    #            #print("\t",name)
    
    return model_ft, params_to_update
        


def parameter_coarse_to_fine_search(iter, dataloader_dict, params_to_update):
 
        ## COARSE SEARCH
        coarse_lr = np.arange(1e-4, 1e-1, 1e-2)
        print(coarse_lr.shape)
        coarse_val_accuracies = []
        for lr in coarse_lr:
            model_ft, params_to_update = initialize_training()
            # Train model with lr
            optimizer_ft = optim.Adam(params_to_update, lr=lr)
            # Setup the loss fxn
            criterion = nn.CrossEntropyLoss()
            # Train and evaluate
            model_ft, hist, _, _ = train_model(model_ft, dataloader_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
            coarse_val_accuracies.append( hist[-1] )
            
        #Plot results
        #plt.scatter(coarse_lr, coarse_val_accuracies)
        #plt.xlabel('lambda')
        #plt.ylabel('val accuracy')
        #plt.savefig('coarse_seach.png')
        #plt.close()
        f = open("coarse.txt", "a")
        for idx, val in enumerate(coarse_val_accuracies):
            f.write(str(coarse_lr[idx])+ ", " + str(val.item()*100)+ "%\n" )
            print("(", coarse_lr[idx], ",",val.item()*100, "% )" )
        f.close()
        
        return (0,0)
        ### FINE RANDOM SEARCH
        best_3_lr = np.take(coarse_lr, np.argsort(coarse_val_accuracies)[-3:])

        l_min, l_max = min(best_3_lr), max(best_3_lr)
        accs = []
        etas = []
        for _ in range(iter):
            lr = l_min + (l_max - l_min) * random.uniform(0,1)
            etas.append(lr)
            model_ft, params_to_update = initialize_training()
            # Train model with lr
            optimizer_ft = optim.Adam(params_to_update, lr=lr)
            # Setup the loss fxn
            criterion = nn.CrossEntropyLoss()
            # Train and evaluate
            model_ft, hist, _, _ = train_model(model_ft, dataloader_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
            accs.append( hist[-1] )

        for idx, val in enumerate(accs):
            print("(", etas[idx], ",",val*100, "% )" )

        best_3_lr = np.take(etas, np.argsort(accs)[-3:])
        print("The 3 best lambda values:", best_3_lr, "and their respective accuracies on validation dataset",np.take(accs, np.argsort(accs)[-3:]))

        #Plot results

        #plt.scatter(etas, accs)
        #plt.xlabel('lr')
        #plt.ylabel('val accuracy')
        #plt.savefig('fine_search.png')
        #plt.close()
        best_found = (np.take(etas, np.argsort(accs)[-1:])[0] , np.take(accs, np.argsort(accs)[-1:])[0])
        return best_found

def pre_process_dataset(input_size, subset = None):
    files = ['./data/oxford-iiit-pet/annotations/test.txt', './data/oxford-iiit-pet/annotations/trainval.txt']
    data = [[] for i in range(len(files))]
    labels = [[] for i in range(len(files))]
    for i in range(len(files)):
        with open(files[i]) as f:
            lines = f.readlines()
            if subset:
                for line in np.random.permutation(lines)[:subset]:
                    label = 0 if line.split(" ")[0][0].isupper() else 1  
                    labels[i].append(label)
                    data[i].append('./data/oxford-iiit-pet/images/'+str(line.split(" ")[0]))
            else:
                for line in lines:
                    label = 0 if line.split(" ")[0][0].isupper() else 1  
                    labels[i].append(label)
                    data[i].append('./data/oxford-iiit-pet/images/'+str(line.split(" ")[0]))


    trainingval_data = CustomDataset(
        split = 'trainval',
        img_paths=data[1],
        labels=labels[1],
        input_size = input_size
    )
    test_data = CustomDataset(
        split = 'test',
        img_paths=data[0],
        labels=labels[0],
        input_size = input_size
    )
    # Load training and validation datasets
    train_dataset, val_dataset = torch.utils.data.random_split(trainingval_data, [int(len(trainingval_data)*0.8),int(len(trainingval_data)*0.2 ) ])
    


    # Create training and validation dataloaders
    dataloaders_dict = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
                        'val':  torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)}
    
    dataloaders_dictest = {'test': torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)}

    return dataloaders_dict, dataloaders_dictest
    

def main():
    # Load pretrained dataset
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    model_ft = model_ft.to(device)
    """
    trainingval_data = torchvision.datasets.OxfordIIITPet(
        root = "data",
        split = 'trainval',
        download = True,
        transform = transforms.Compose([
            # TODO: kolla om det är rätt transforms
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
            # TODO
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )
    """
    """
    data = []
    labels = []
    data_dir = glob(".\data\oxford-iiit-pet\images\*.jpg")
    # Load the images and get the classnames from the image path
    for image in data_dir:
        name = image.rsplit("\\",1)[1]
        if name[0].isupper() == True:
            label = 1
        else: 
            label = 2
        data.append(image)
        labels.append(label)
    """
    #print(dataloaders_dict['train'])
    #train_features, train_labels = next(iter(dataloaders_dict["train"]))
    #val_features, val_labels = next(iter(dataloaders_dict["val"]))
    """

    id_to_specie = {}
    with open('data/oxford-iiit-pet/annotations/list.txt') as f:
        lines = f.readlines()
        for _, line in enumerate(lines[6:],6):
            tokens = line.split(' ')
            id_to_specie[str(int(tokens[1])-1)] = int(tokens[2])-1
    
    # Convert Class id to specie ID in train and val datasets
    for idx, lab in enumerate(train_labels):
        train_labels[idx]=(id_to_specie[str(lab.item())])
    
    for idx, lab in enumerate(val_labels):
        val_labels[idx]=(id_to_specie[str(lab.item())])
    """
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():

            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    dataloaders_dict, dataloaders_dictest = pre_process_dataset(input_size=input_size, subset=1840)

    ### Learning rate search:
    best_lr = parameter_coarse_to_fine_search(20, model_ft, dataloaders_dict, params_to_update)
    
    print("best_lr", best_lr)

    """
    # Observe that all parameters are being optimized
    ## SGD
    #optimizer_ft = optim.Adam(params_to_update, lr=0.001, momentum=0.9)
    ## Adam
    optimizer_ft = optim.Adam(params_to_update, lr=best_lr[0])

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist, train_loss_hist, val_loss_hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
    
    test_hist = test_model(model_ft, dataloaders_dictest)
    print(test_hist)
    # Initialize the non-pretrained version of the model used for this run
    """
    scratch_model,_ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)
    scratch_model = scratch_model.to(device)
    scratch_optimizer = optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9)
    scratch_criterion = nn.CrossEntropyLoss()
    _,scratch_hist = train_model(scratch_model, dataloaders_dict, scratch_criterion, scratch_optimizer, num_epochs=num_epochs, is_inception=(model_name=="inception"))
    """
    # Plot the training curves of validation accuracy vs. number
    #  of training epochs for the transfer learning method and
    #  the model trained from scratch
    ohist = []
    shist = []

    #ohist = [h.cpu().numpy() for h in hist]
    #shist = [h.cpu().numpy() for h in test_hist]
    plt.plot(val_loss_hist)
    plt.plot(train_loss_hist)
    plt.savefig("train_val_loss.png")
    plt.show()
    """
    """
    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1,num_epochs+1),ohist,label="Pretrained")
    plt.plot(range(1,num_epochs+1),shit,label="Test")
    plt.ylim((0,1.))
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    plt.show()
    """


    #print(f"Feature batch shape: {train_features.size()}")
    #print(f"Labels batch shape: {train_labels.size()}")

    #img = train_features[0].squeeze()
    #label = train_labels[0]
    #print(f"Label: {label}")
    return 0


if __name__ == '__main__':
    main()
