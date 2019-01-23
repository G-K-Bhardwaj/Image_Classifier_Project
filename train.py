# Imports here
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from collections import OrderedDict

import numpy as np
import time
import argparse

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, help='Path to dataset ')
parser.add_argument('--save_dir', type=str, help='Path to directory to save the checkpoint file.')
parser.add_argument('--arch', type=str, help='Model architecture')
parser.add_argument('--learning_rate', type=float, help='Learning rate')
parser.add_argument('--hidden_units', type=int, help='Number of hidden units')
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
#parser.add_argument('--checkpoint', type=str, help='Save trained model checkpoint to file')

args, _ = parser.parse_known_args()

# Use command line values when specified
if args.data_dir:
    data_dir = args.data_dir
else:
    data_dir = 'flowers'
    
if args.save_dir:
    save_dir = args.save_dir

if args.arch:
    arch = args.arch
        
if args.learning_rate:
    learning_rate = args.learning_rate
else:
    learning_rate = 0.001

if args.hidden_units:
    hidden_units = args.hidden_units
else:
    hidden_units = 520

if args.epochs:
    epochs = args.epochs
else:
    epochs = 1

if args.gpu:
    gpu = args.gpu
else:
    gpu = False


train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# TODO: Define your transforms for the training, validation, and testing sets
normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])

train_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomRotation(30),
                                       transforms.ToTensor(),
                                       normalize])

test_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       normalize])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       normalize])

# TODO: Load the datasets with ImageFolder
image_datasets = {'train': datasets.ImageFolder(train_dir, transform = train_transforms),
                  'test': datasets.ImageFolder(test_dir, transform = test_transforms),
                  'valid': datasets.ImageFolder(valid_dir, transform = valid_transforms)
                }

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True)



# TODO: Build model
def build_model(model_name, hidden_size, drop = 0.5):
    try:
        if model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
            input_size = 25088
        elif model_name == 'densenet121':
            model = models.densenet121(pretrained=True)
            input_size = 1024
        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
            input_size = 9216
        else:
            raise ValueError('This model is not supported, Please choose a model from "vgg16", "densenet121" or "alexnet"')
    except ValueError:
        raise
        
    
    print('building model: ', model_name);
        
    for param in model.parameters():
        param.requires_grad = False
        
    print("Input size: ", input_size)
    output_size = 102
    
    classifier = nn.Sequential(OrderedDict([                                            
                                                ('fc1', nn.Linear(input_size, hidden_size)),
                                                ('relu1', nn.ReLU()),
                                                ('dropout1', nn.Dropout(p=drop)),
                                                ('fc2', nn.Linear(hidden_size, output_size)),
                                                ('output', nn.LogSoftmax(dim=1))
                                                ]))
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = classifier
    return model


# TODO: Train your network
def train(model, epochs, print_every, learning_rate, criterion, optimizer, trainloader, validloader, gpu=False):
    steps = 0
    
    # Turn on Cuda if available
    if gpu:
        if torch.cuda.is_available():
            model.to('cuda')
        else:
            print("Cuda is not available using cpu.")
    
    for e in range(epochs):
        running_loss = 0
        e_tm = time.time();
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            
            if gpu and torch.cuda.is_available():
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            optimizer.zero_grad()
            
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                valid_accuracy=0

                # Iterate through images/labels for validation
                for i, (valid_inputs,valid_labels) in enumerate(validloader):
                    optimizer.zero_grad()

                    if gpu and torch.cuda.is_available():
                        valid_inputs, valid_labels = valid_inputs.to('cuda') , valid_labels.to('cuda')
                        model.to('cuda')

                    with torch.no_grad():    
                        outputs = model.forward(valid_inputs)
                        valid_loss = criterion(outputs,valid_labels)
                        ps = torch.exp(outputs)
                        equality = (valid_labels.data == ps.max(1)[1])
                        valid_accuracy += equality.type_as(torch.FloatTensor()).mean()

                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Loss: {:.4f}...".format(running_loss/print_every),
                      "Validation Loss: {:.4f}...".format(valid_loss / len(validloader)),
                      "Validation Accuracy: {:.4f}...".format(valid_accuracy /len(validloader))
                     )
                
                running_loss = 0
                
        print("Epoch runtime: {}...".format(time.time()- e_tm))
    return model

def check_accuracy_on_test(testloader, gpu=False):
    correct = 0
    total = 0
    
    # change to cuda if available
    if gpu and torch.cuda.is_available():
        model.to('cuda')
        
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

def save_model(save_dir, model_name, hidden_units=512):
    checkpoint = {'model': model_name,
                  'hidden_layers': hidden_units,
                  'state_dict': model.state_dict(),
                  'epochs': epochs+1,
                  'optimizer': optimizer.state_dict(),
                  'class_index': image_datasets['train'].class_to_idx
                  }
    torch.save(checkpoint, save_dir)


# Train model if invoked from command line
if args.arch:
    model = build_model(arch, hidden_units, drop=0.5)
    print_every = 10
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model = train(model, epochs, print_every, learning_rate, criterion, optimizer, trainloader, validloader, gpu)
    check_accuracy_on_test(testloader, gpu)

    if args.save_dir:
        save_model(save_dir, arch, hidden_units=512)

