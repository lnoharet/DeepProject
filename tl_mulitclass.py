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

seed_ = 0.444
torch.manual_seed(seed_)
torch.cuda.manual_seed_all(seed_)
torch.backends.cudnn.deterministic = True

""" Runnning Options """
PARAM_SEARCH = True
LOAD_SAVE = False
SCHEDULE = None #'1cycle' # ExpLR

# Top level data directory.
data_dir = "./data/oxford-iiit-pet"
DATA_SUBSET = None # None = whole dataset
default_lr = 0.0001 # best lr for FC layer


BN = False # false = exclude BN params from fine tuning
ft_layers = 2 # idx 1-5
layers = ["fc", 'layer4', 'layer3', 'layer2', 'layer1']
parameter_search_layer = 'fc' # for parameter search
lr_1 = 0
lr_2 = 0
lr_3 = 1e-7
lr_4 = 2.9e-6
lr_fc = 0.0001

""" SEARCH PARAMS """

#coarse_lr = np.array([ 1e-6, 1e-7, 1e-8, 1e-9 ])#, 0.0000095, 0.00001, 0.000015, 0.00002, 0.000025, 0.00003, 0.000035, 0.00004])

#coarse_lr = np.array([0.00001,0.00002,0.00003,0.00004,0.00005,0.00006,0.00007,0.00008,0.00009])
#coarse_lr = np.array([0.0009, 0.0095])

l_max = 1.1e-6
l_min = 1e-07
coarse_lr = []
for i in range(0,10):
    lr = l_min + (l_max-l_min)*random.uniform(0,1)
    coarse_lr.append(lr)
coarse_lr = np.array(coarse_lr)



# Models from [resnet18, resnet34]
model_name = "resnet18"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameters
num_classes = 37
batch_size = 8
num_epochs = 15

class CustomDataset(Dataset):
    def __init__(self, img_paths, labels, input_size, split):
            
        self.img_labels = labels
        self.img_paths = img_paths

        if split == 'trainval':
            self.transform = transforms.Compose([
                # TODO: kolla om det är rätt transforms
                #transforms.RandomResizedCrop(input_size),
                transforms.Resize((input_size, input_size)),
                #transforms.CenterCrop(input_size),
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

def load_image(filename) :
    img = Image.open(filename)
    img = img.convert('RGB')
    return img

def freeze_all_params(model, params_list):
    for name,param in model.named_parameters():
        if name not in params_list:
            param.requires_grad = False
 
def train_model(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=25, is_inception=False, used_lr = None):
    since = time.time()

    val_acc_history = []
    train_loss_history = []
    val_loss_history = [] 
    train_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    lrs = []

    for epoch in range(num_epochs):
        if PARAM_SEARCH:
            print('Epoch {}/{} lr = {}'.format(epoch, num_epochs - 1, used_lr))
        else:
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
            for inputs, labels in dataloaders[phase]:
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
                        lrs.append(optimizer.param_groups[0]["lr"]) # track learning rate
                        if scheduler and SCHEDULE == '1cycle':
                            scheduler.step()


                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if scheduler and phase == 'val' and SCHEDULE == 'ExpLR':
                scheduler.step()

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
                train_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_acc_history, val_acc_history, train_loss_history, val_loss_history, lrs


def test_model(model, dataloaders):
    """ Evaluates the model on the test data. Returns test accuracy"""
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


def initialize_model(model_name, num_classes, fc_lr = lr_fc, lay4_lr = lr_4, lay3_lr =lr_3,lay2_lr = lr_2, lay1_lr =lr_1, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18 """
        model_ft = models.resnet18(pretrained=use_pretrained)
    elif model_name == "resnet34":
        """ Resnet34 """
        model_ft = models.resnet34(pretrained=use_pretrained)

    """ Set layers to be fine-tuned """
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224
    params_to_update = []
    all_layers =[{"params": model_ft.layer1.parameters(), "lr":lay1_lr},{"params": model_ft.layer2.parameters(), "lr":lay2_lr},{"params": model_ft.layer3.parameters(), "lr":lay3_lr},{"params": model_ft.layer4.parameters(), "lr":lay4_lr}, {"params": model_ft.fc.parameters(), "lr":fc_lr}]
    for l in reversed(range(ft_layers)):
        params_to_update.append(all_layers[l])
        

    #params_to_update = [{"params": model_ft.layer4.parameters(), "lr":lay4_lr}, {"params": model_ft.fc.parameters(), "lr":fc_lr}]
                        

    model_ft = model_ft.to(device)
    params_to_list = []
    for name, param in model_ft.named_parameters():
        for l in range(ft_layers):
            if layers[l] in name: 
                if not BN:
                    if "bn" not in name:
                        params_to_list.append(name)
                else:
                    params_to_list.append(name)
    freeze_all_params(model_ft, params_to_list)
    #params_to_update = []
    #    for name,param in model_ft.named_parameters():
    #    if param.requires_grad == True:
    #        params_to_update.append(param)"""
    

    return model_ft, input_size, params_to_update
        
def plot_parameter_search(params, accs, ):
    # TODO
    #Plot results
    #plt.scatter(coarse_lr, coarse_val_accuracies)
    #plt.xlabel('lambda')
    #plt.ylabel('val accuracy')
    #plt.savefig('bin_plots/coarse_seach' + str(round(time.time())) +'.png')
    #plt.close()
    return

def plot(train, val, mode, used_lr, test_acc):
    plt.plot(val, label='val')
    plt.plot(train, label='train')
    plt.xlabel('epoch')
    plt.ylabel(mode)
    plt.title(mode + ' with lr=' + str(used_lr) + ' n_batch=' + str(batch_size) + ' test_acc=' + str(test_acc))
    plt.legend()
    if mode == "acc":  plt.ylim([0, 1])
    plt.savefig('mul_plots/' + mode + str(round(time.time()) - 1650000000) + '.png')
    plt.close()
    return

def plot_lrs(lrs):
    plt.plot(lrs)
    plt.xlabel('epoch')
    plt.ylabel('learning rate')
    plt.title('Learning rate throughout training')
    plt.savefig('mul_plots/' + 'eta' + str(round(time.time()) - 1650000000) + '.png')
    plt.close()
    return


def parameter_search(dataloader_dict, params_to_update, test_data):
 
        ## COARSE SEARCH
        print('--- PARAMETER SEARCH ---')
        print('Searched parameters:', coarse_lr, 'for layer', parameter_search_layer)
 
        val_accuracies = [] 
        test_accs = []


        for lr in coarse_lr:
            if parameter_search_layer == 'fc':
                model_ft, _, params_to_update = initialize_model(model_name, num_classes, fc_lr=lr)
            elif parameter_search_layer == '1':
                model_ft, _, params_to_update = initialize_model(model_name, num_classes, lay1_lr=lr)
            elif parameter_search_layer == '2':
                model_ft, _, params_to_update = initialize_model(model_name, num_classes, lay2_lr=lr)
            elif parameter_search_layer == '3':
                model_ft, _, params_to_update = initialize_model(model_name, num_classes, lay3_lr=lr)
            elif parameter_search_layer == '4':
                model_ft, _, params_to_update = initialize_model(model_name, num_classes, lay4_lr=lr)

            # Train model with lr
            optimizer_ft = optim.Adam(params_to_update, lr=lr)
            # Setup the loss fxn
            criterion = nn.CrossEntropyLoss()
            # Train and evaluate
            model_ft, train_hist, hist, train_loss, val_loss = train_model(model_ft, dataloader_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"), used_lr = lr)
            
            test_acc = test_model(model_ft, test_data)[-1].item()*100
            test_accs.append(test_acc)
            print('test acc for lr={} is {} %'.format(lr, test_acc))
            plot(train_loss, val_loss, 'loss', lr, test_acc)
            plot(train_hist, hist, 'acc', lr, test_acc)

            val_accuracies.append( hist[-1] )
            

        # writes coarse results to txt file
        f = open("mul_plots/coarse.txt", "a")
        f.write('\n')
        for idx, val in enumerate(val_accuracies):
            f.write(str(coarse_lr[idx])+ ", " + str(val.item()*100)+ "%\n" )
            print("lr=",coarse_lr[idx], ",val =",val.item()*100, "% | test=",test_accs[idx],'%' )
        f.close()

        plot_parameter_search(coarse_lr, val_accuracies )
        best_found = np.take(coarse_lr, np.argsort(test_accs)[-1:])

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
                    label = int(line.split(" ")[1]) - 1   
                    labels[i].append(label)
                    data[i].append('./data/oxford-iiit-pet/images/'+str(line.split(" ")[0]))
            else:
                for line in lines:
                    label = int(line.split(" ")[1]) - 1  
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

def download_data():
    trainingval_data = torchvision.datasets.OxfordIIITPet(
    root = "data",
    download = True
    )


def main():
    print("Initializing model")
    if LOAD_SAVE:
        model_ft = torch.load("model.pt")
        model_ft.eval()
    else:    
        model_ft, input_size, params_to_update = initialize_model(model_name, num_classes, use_pretrained=True)

    # Load pretrained model
    # Load pretrained model
    #print(model_ft)
    #downl oad_data()

    # Print the params we fine-tune
    print("Params to learn:")
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)


    # Change labels of data to be binary for specie classification
    trainval_data, test_data = pre_process_dataset(input_size=input_size, subset=DATA_SUBSET)

    if PARAM_SEARCH:
        ### Learning rate search:
        best_lr = parameter_search(trainval_data, params_to_update, test_data)
        print("Parameter search yielded best test acc with lr =", best_lr[0])
        used_lr = best_lr[0]
    else:
        used_lr = default_lr

        if not LOAD_SAVE: 
            ## Adam
            optimizer_ft = optim.Adam(params_to_update)#, lr=used_lr)
            if SCHEDULE == '1cycles':
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer_ft, 0.1, total_steps=None, epochs=num_epochs, steps_per_epoch=int(3680/batch_size), pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0, three_phase=False, last_epoch=- 1, verbose=False)
            if SCHEDULE == 'ExpLR':
                scheduler =  torch.optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma=0.05, verbose= True)

            

            # Setup the loss fxn
            criterion = nn.CrossEntropyLoss()

            # Train and evaluate
            print('--- Training with adam ---')
            model_ft, train_hist, hist, train_loss_hist, val_loss_hist = train_model(model_ft, trainval_data, criterion, optimizer_ft, scheduler=scheduler,num_epochs=num_epochs, is_inception=(model_name=="inception"))
            
            plot(train_loss_hist, val_loss_hist, "loss", used_lr, round(test_acc,4))
            plot(train_hist, hist, "acc", used_lr, round(test_acc,4))

        # Eval model on test data
        print('--- Testing model on testdata ---')
        test_acc = test_model(model_ft, test_data)[-1].item()*100
        print("Test Acc = ", test_acc)

        if LOAD_SAVE: 
            torch.save(model_ft, "model.pt")
            print("saved")









    """ BASELINE """
    ## Calculate baseline for comparison
    # Initialize the non-pretrained version of the model used for this run
    """
    scratch_model,_ = initialize_model(model_name, num_classes, use_pretrained=False)
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

    return 0


if __name__ == '__main__':
    main()

#### DICT ID to SPECIE
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

