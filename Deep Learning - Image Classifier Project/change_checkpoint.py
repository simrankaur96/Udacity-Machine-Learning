import torch
from torch import nn, optim
import torch.nn.functional as F

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from collections import OrderedDict

from torchvision import datasets, transforms, models

def make_model(input_size, hidden_size, output_size, dropout_prob=0.2):
    
    model = models.vgg16(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(input_size, hidden_size[0])),
                                ('relu', nn.ReLU()),
                                ('dropout', nn.Dropout(p=dropout_prob)),
                                ('fc2', nn.Linear(hidden_size[0], hidden_size[1])),
                                ('relu2', nn.ReLU()),
                                ('dropout2', nn.Dropout(p=dropout_prob)),
                                ('fc3', nn.Linear(hidden_size[1], output_size)),
                                ('out', nn.LogSoftmax(dim=1))
                                ]))
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.02)

    return model, criterion, optimizer

def load_model(path):
    checkpoint = torch.load(path)
    model,_,_ = make_model(input_size = checkpoint["input_size"], 
                           hidden_size = checkpoint["hidden_size"], 
                           output_size = checkpoint["output_size"],
                           dropout_prob = 0.2)
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model
    
model = load_model('checkpoint2.pth')
print(model)

model.class_to_idx = datasets.ImageFolder('./flowers/train').class_to_idx
model.cpu
torch.save({'input_size': 25088,
            'hidden_size': 1024,
            'dropout': 0.2,
            'lr': 0.02,
            'no_of_epochs': 10,
            'output_size': 102,
            'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx},
            'checkpoint.pth')