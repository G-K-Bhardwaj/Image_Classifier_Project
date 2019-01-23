import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from collections import OrderedDict

import PIL
from PIL import Image
from PIL import ImageFile
import json

import numpy as np
import argparse

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='Image to predict')
parser.add_argument('checkpoint', type=str, help='Model checkpoint to use when predicting')
parser.add_argument('--category_names', type=str, help='JSON file containing label names', default='')
parser.add_argument('--top_k', type=int, help='Return top K predictions', default=5)
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

args, _ = parser.parse_known_args()

print(args)

# Use command line values when specified
if args.input:
    image = args.input   
    
if args.checkpoint:
    checkpoint = args.checkpoint

if args.top_k:
    topk = args.top_k
        
if args.category_names:
    category_names = args.category_names
else:
    category_names = ''

if args.gpu:
    gpu = args.gpu
else:
    gpu = False

# TODO: Build model
def build_model(model_name, hidden_size, drop = 0.5):
    
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
        print('choose a model "vgg16", "densenet121" or "alexnet"')

    
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

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_rebuild(file_dir):
    # Load saved pre-trained model
    checkpoint = torch.load(file_dir, map_location=lambda storage, loc: storage)
    
    hidden_size = checkpoint['hidden_layers']
    drop_out = 0.35
    
    # Get model name
    if checkpoint['model'] == 'vgg16':
        model_name = 'vgg16'
    elif checkpoint['model'] == 'densenet121':
        model_name = 'densenet121'
    elif checkpoint['model'] == 'alexnet':
        model_name = 'alexnet'
    else:
        print("Please choose a model from 'vgg16', 'densenet121', 'alexnet'.")

    # build model
    model = build_model(model_name, hidden_size, drop = drop_out)    
    
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.class_to_idx = checkpoint['class_index']
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    base_size = 256
    img = Image.open(image)
    width, height = img.size
    
    if width < height:
        img.thumbnail((base_size, height)) 
    else:
        img.thumbnail((width, base_size))
        
    width, height = img.size
        
    # crop center 224*224
    new_size = [224, 224]
        
    left = width//2 - new_size[0]//2
    upper = height//2 - new_size[1]//2
    right = width//2 + new_size[0]//2
    lower = height//2 + new_size[1]//2
    
    crop_box = [left, upper, right, lower]
    img = img.crop(crop_box)
    img.load()
    
    np_image = np.array(img)
    
    #convert to tensor to normalise
    to_tensor = transforms.ToTensor()
    img = to_tensor(np_image)
    
    # normalize 
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])
    
    img = normalize(img)
    img = np.array(img)
    
    return img


def predict(img, model, category_names='', topk=5, gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    image = process_image(img)
    image = Variable(torch.FloatTensor(image), requires_grad=True)
    image = image.unsqueeze(0)
    
    # Turn on Cuda if available
    if gpu:
        if torch.cuda.is_available():
            image = image.cuda()
            model.to('cuda')
        else:
            print("Cuda is not available using cpu.")

    # Turn on evaluation for model
    model.eval()

    with torch.no_grad(): 
        output = model.forward(image) 
        ps = torch.exp(output)
        
        idx_to_class = {v: k for k, v in model.class_to_idx.items()} 
        probs, idx = ps.topk(topk)
    
    classes = []
    for i in idx[0].cpu().numpy().tolist():
        classes.append(idx_to_class[i])

    labels = []
    if len(category_names)>0:
        # load labels
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        
        for flower_idx in classes:
            labels.append(cat_to_name[flower_idx])
    
        classes = labels
    return probs[0].cpu().numpy(), classes
    

# Perform predictions if invoked from command line
if image and checkpoint:
    checkpoint_path = checkpoint;
    model = load_rebuild(checkpoint_path)
    probs, classes = predict(image, model, category_names, topk, gpu)
    print(probs)
    print(classes)





