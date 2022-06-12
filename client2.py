import numpy as np
import pandas as pd

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from torch import Tensor


import matplotlib.pyplot as plt
import time
import copy
from random import shuffle

import tqdm as tqdm

import sklearn
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.metrics import classification_report
from PIL import Image
import cv2

import os
import shutil

from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch


import flwr as fl


from typing import Tuple, Dict

from random import random, randint, sample, choice






DATA_PATH = 'dataset/dataset'


COVID_PATH = 'dataset/dataset/covid19'
NORMAL_PATH = 'dataset/dataset/normal'
#CHANGE YOUR LOCATION ACCORDING TO YOUR FILE PATH

# DATA_PATH = './dataset'
# COVID_PATH = './dataset/covid19'
# NORMAL_PATH = './dataset/normal'


class_names = os.listdir(DATA_PATH)
image_count = {}
for i in class_names:
    image_count[i] = len(os.listdir(os.path.join(DATA_PATH,i)))

def load_data(datadir,  valid_size = .2) -> Tuple[
    torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict
]:
    """Load chest X-ray images (training and test set)."""
    mean_nums = [0.485, 0.456, 0.406]
    std_nums = [0.229, 0.224, 0.225]

    data_transforms = {"train":transforms.Compose([
                                transforms.Resize((150,150)), #Resizes all images into same dimension
                                transforms.RandomRotation(10), # Rotates the images upto Max of 10 Degrees
                                transforms.RandomHorizontalFlip(p=0.4), #Performs Horizantal Flip over images 
                                transforms.ToTensor(), # Coverts into Tensors
                                transforms.Normalize(mean = mean_nums, std=std_nums)]), # Normalizes
                    "val": transforms.Compose([
                                transforms.Resize((150,150)),
                                transforms.CenterCrop(150), #Performs Crop at Center and resizes it to 150x150
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean_nums, std = std_nums)
                    ])}
    
    train_data = datasets.ImageFolder(datadir,       
                    transform=data_transforms['train']) #Picks up Image Paths from its respective folders and label them
    test_data = datasets.ImageFolder(datadir,
                    transform=data_transforms['val'])
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    dataset_size = {"train":len(train_idx), "val":len(test_idx)}
    train_sampler = SubsetRandomSampler(train_idx) # Sampler for splitting train and val images
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=8) # DataLoader provides data from traininng and validation in batch of 8
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=8)
    return trainloader, testloader, dataset_size
    
trainloader, testloader, dataset_size =load_data(DATA_PATH,0.2)
dataloaders = {"train":trainloader, "val":testloader}
data_sizes = {x: len(dataloaders[x].sampler) for x in ['train','val']}
class_names = trainloader.dataset.classes


if torch.cuda.is_available():  #PyTorch has device object to load the data into the either of two hardware [CPU or CUDA(GPU)]
    DEVICE=torch.device("cuda:0")
    print("Training on GPU")
else:
    DEVICE = torch.device("cpu")
    print("Training on CPU")



class Net(nn.Module):
    def CNN_Model(pretrained=True):
        model = models.densenet121(pretrained=pretrained) # Returns Defined Densenet model with weights trained on ImageNet
        num_ftrs = model.classifier.in_features # Get the number of features output from CNN layer
        model.classifier = nn.Linear(num_ftrs, len(class_names)) # Overwrites the Classifier layer with custom defined layer for transfer learning
        model = model.to(DEVICE) # Transfer the Model to GPU if available
        return model
    

model = Net.CNN_Model(pretrained=True)

# specify loss function (categorical cross-entropy loss)
criterion = nn.CrossEntropyLoss() 

# Specify optimizer which performs Gradient Descent
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(
    net: Net,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,  # pylint: disable=no-member
) -> None:
    """Train the network."""

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")

    # Train the network
    
    since = time.time()


    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            current_loss = 0.0
            current_corrects = 0
            
            for inputs, labels in tqdm.tqdm(dataloaders['train'], desc=phase, leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # We need to zero the gradients in the Cache.
                optimizer.zero_grad()

                # Time to carry out the forward training poss
                # We only need to log the loss stats if we are in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # We want variables to hold the loss statistics
                current_loss += loss.item() * inputs.size(0)
                current_corrects += torch.sum(preds == labels.data)
                
                
            epoch_loss = current_loss / data_sizes['train']
            epoch_acc = current_corrects.double() / data_sizes['train']
            print('{} Loss: {:.4f} | {} Accuracy: {:.4f}'.format(
                    phase, epoch_loss, phase, epoch_acc))

            

    
    time_since = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_since // 60, time_since % 60))
    


                

                
                

def test(
    net: Net,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,  # pylint: disable=no-member
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    # Define loss and metrics
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0

    # Evaluate the network
    net.to(device)
    net.eval()
    
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['train']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy

class Covid_Fed_Client(fl.client.NumPyClient):
    """Flower client implementing covid-19 image classification using
    PyTorch."""

    def __init__(
        self,
        model: Net(),
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        data_sizes: Dict,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.data_sizes = data_sizes
        #return the model weight as a list of NumPy ndarrays
    def get_parameters(self) -> List[np.ndarray]: 
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    
    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # set the local model weights
        # train the local model
        # receive the updated local model weights
        self.set_parameters(parameters)
        train(self.model, self.trainloader, epochs=1, device=DEVICE)
        return self.get_parameters(), self.data_sizes["train"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader, device=DEVICE)
        return float(loss), self.data_sizes["train"], {"accuracy": float(accuracy)}


def main() -> None:
    """Load data, from local directory."""

    # Load model and data
    model = Net()
    model.to(DEVICE)
    trainloader, testloader, num_examples = load_data(DATA_PATH,0.2)
    print("Start training")
#     train(net=Net(), trainloader=trainloader, epochs=1, device=DEVICE)
    print("Evaluate model")
    loss, accuracy = test(net=Net(), testloader=testloader, device=DEVICE)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    # created an instance of our class Covid_Fed_Client and add one line to actually run this client:
    client = Covid_Fed_Client(model, trainloader, testloader, num_examples)
    fl.client.start_numpy_client("localhost:8080", client)


if __name__ == "__main__":
    main()


