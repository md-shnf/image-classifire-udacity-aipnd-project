import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from torch.autograd import Variable
from PIL import Image
import matplotlib.image as mpimg
import seaborn as sns
import json
import argparse
from matplotlib.ticker import FormatStrFormatter




def main():
    parser = argparse.ArgumentParser(description='Flower image classification (trainer)')
    parser.add_argument('--gpu', type=bool, default=False, help='Use GPU if its available')
    parser.add_argument('--arch', type=str, default='densenet', help='arch Model', required=True)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=100, help='hidden units ')
    parser.add_argument('--epochs', type=int, default=15, help='number of epochs')
    parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
    parser.add_argument('--saved_model' , type=str, default='checkpoint.pth', help='path of your saved model')
    args = parser.parse_args()
    
    import json
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

     train_model(args)
     validate_model(args)   
if __name__ == "__main__":
    main()

def form_data(args):

    data_dir = data_dir + 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'train' : transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),transforms.RandomRotation(30),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                             [0.229, 0.224, 0.225])]),

        'valid' : transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])]),

        'test' : transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    }


    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        x: datasets.ImageFolder(root=data_dir + '/' + x, transform=data_transforms[x])
        for x in list(data_transforms.keys())
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    data_loader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True)
                  for x in ['train', 'valid', 'test']}

    dataset_size = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    class_to_idx = image_datasets['train'].class_to_idx

    return dataloaders, image_datasets


def train_model():


    model.train()
    epochs = 3
    print_every = 40
    steps = 0
    use_gpu = False
    
    # Use command line values when specified
    if args.arch:
        arch = args.arch     
        
    if args.hidden_units:
        hidden_units = args.hidden_units

    if args.epochs:
        epochs = args.epochs
            
    if args.learning_rate:
        learning_rate = args.learning_rate

    if args.gpu:
        gpu = args.gpu

    if args.checkpoint:
        checkpoint = args.checkpoint   
        
    if torch.cuda.is_available():
        use_gpu = True
        model.to('cuda')
    else:
        model.cpu()

    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in iter(data_loader['train']):
            steps += 1
            
            
            if use_gpu:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
            else:
                inputs = Variable(inputs)
                labels = Variable(labels) 

           # inputs,labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]

            if steps % print_every == 0:
                validation_loss, accuracy = validate_model()
                
                print("Epoch: {}/{} ".format(epoch+1, epochs),
                        "Training Loss: {:.4f} ".format(running_loss/print_every),
                        "Validation Loss: {:.4f} ".format(validation_loss),
                        "Validation Accuracy: {:.4f}".format(accuracy)) 
       return model         
                
def validate_model():
    model.eval()
    accuracy = 0
    test_loss = 0
    
    for inputs, labels in iter(data_loader['valid']):
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
            labels = labels.to('cuda') 
        else:
            inputs = Variable(inputs)
            labels = Variable(labels) 

        output = model.forward(inputs)
        test_loss += criterion(output, labels).data[0]
        ps = torch.exp(output).data 
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()
        
    return test_loss/len(data_loader['valid']), accuracy/len(data_loader['valid'])

    

