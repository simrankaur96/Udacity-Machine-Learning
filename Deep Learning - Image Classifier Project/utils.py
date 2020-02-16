#importing all header
import torch
from torch import nn, optim
import torch.nn.functional as F

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from collections import OrderedDict

from torchvision import datasets, transforms, models

structure = {"vgg16":25088,
             "densenet121":1024,
             "vgg13":25088,
             "resnet18":512}

def data_load(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #Define your transforms for the training, validation, and testing sets
    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(225),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.465, 0.406], [0.229, 0.224, 0.225])])

    
    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders 
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=32)
    
    print("Data is loaded onto trainloader, validloader and testloader from ",data_dir)
    return trainloader, validloader, testloader

def create_network(arch = 'vgg16', learning_rate=0.02, hidden_units=1024, dropout_prob = 0.2):
    
    
    if(arch == 'vgg16'):
        model = models.vgg16(pretrained=True)
    elif(arch == 'vgg13'):
        model = models.vgg13(pretrained=True)
    elif(arch == 'resnet18'):
        model = models.resnet18(pretrained=True)
    elif(arch == 'densenet121'):
        model = models.densenet121(pretrained=True)
    else:
        print("Please choose one of the following archictectures: [ vgg16 | vgg13 | resnet18 | densenet121]")
        
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(structure[arch], hidden_units)),
                                ('relu', nn.ReLU()),
                                ('dropout', nn.Dropout(p=dropout_prob)),
                                ('fc2', nn.Linear(hidden_units, 256)),
                                ('relu2', nn.ReLU()),
                                ('dropout2', nn.Dropout(p=dropout_prob)),
                                ('fc3', nn.Linear(256, 102)),
                                ('out', nn.LogSoftmax(dim=1))
                                ]))
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate)
    
    print("Model is created over {} pretrained model.".format(arch))
    return model, criterion, optimizer

def train_network(model, criterion, optimizer, trainloader, validloader, epochs, gpu):
    if(gpu == 'gpu'):
        device = 'cuda'
    else:
        device = 'cpu'
    
    model.to(device)
    
    steps = 1
    print_every = 10 #print status every 10 steps
    running_loss = 0
    print("--------------------Training the Network------------------------")
    for e in range(epochs):
        for idx, (images, labels) in enumerate(trainloader):
            steps += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:

                model.eval()
                testing_loss = 0
                accuracy = 0

                with torch.no_grad():
                    for images, labels in validloader:

                        images, labels = images.to(device), labels.to(device)
                        out_ps = model.forward(images)
                        testing_loss += criterion(out_ps, labels).item()

                        output = torch.exp(out_ps)
                        equality = output.max(1)[1] == labels
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                print("Epochs {}/{}...".format(e+1, epochs),
                      "Training Loss: {:.3f}..".format(running_loss/print_every),
                      "Cross Validation Testing Loss: {:.3f}..".format(testing_loss/len(validloader)),
                      "Accuracy: {:.3f}".format(accuracy/len(validloader)))
                model.train()
                running_loss = 0
                break
    print("------------------------Finished training the network----------------------------")
    return model    

def save_checkpoint(model, data_dir, save_dir, arch, hidden_size, dropout_prob, lr, epochs):
    model.class_to_idx = datasets.ImageFolder(data_dir+'/train').class_to_idx
    model.cpu
    torch.save({'input_size': structure[arch],
                'hidden_size': hidden_size,
                'dropout': dropout_prob,
                'lr': lr,
                'no_of_epochs': epochs,
                'output_size': 102,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx},
                 save_dir)
    print("Saved the model to {}.".format(save_dir))

def load_checkpoint(path):
    checkpoint = torch.load(path)
    model,_,_ = create_network(learning_rate=checkpoint['lr'], hidden_units=checkpoint['hidden_size'], dropout_prob =                                                   checkpoint['dropout'])
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def predict(pathToImage, model, topk, gpu):
    if(gpu == 'gpu'):
        device = 'cuda'
    else:
        device = 'cpu'
    
    model.to(device)
    img_torch = process_image(pathToImage)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        output = model.forward(img_torch.to(device))
        
    probability = F.softmax(output.data,dim=1)
    probabilities = probability.topk(topk)
    
    return probabilities

def process_image(img_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_pil = Image.open(img_path)
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = adjustments(img_pil)
    
    return img_tensor
    
