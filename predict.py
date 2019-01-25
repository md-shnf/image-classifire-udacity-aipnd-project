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
    
    parser = argparse.ArgumentParser(description='Flower image classification (predictor)')
    parser.add_argument('--gpu', type=bool, default=False, help='Use GPU if its available')
    parser.add_argument('--image_path', type=str, help='path of image')
    parser.add_argument('--saved_model' , type=str, default='checkpoint.pth', help='path of your saved model')
    parser.add_argument('--mapper_json' , type=str, default='cat_to_name.json', help='path of your mapper from category to name')
    parser.add_argument('--topk', type=int, default=5, help='display top k probabilities')

    args = parser.parse_args()

        import json
    with open(args.mapper_json, 'r') as f:
        cat_to_name = json.load(f)
        
        
     model, class_to_idx, idx_to_class = load_model(args)
     top_probs, top_labels, top_flowers =  predict(image, model, top_num=5)
        print("I am done :*(")
        
if __name__ == "__main__":
    main()
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # from mr mukhtar video 
    # TODO: Process a PIL image for use in a PyTorch model
    
    img_loader = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
    
    pil_image = Image.open(image)
    pil_image = img_loader(pil_image).float()
    
    np_image = np.array(pil_image)    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
            
    return np_image


def load_model(model):
    
    model.class_to_idx = image_datasets['train'].class_to_idx
    
    checkpoint = {'model_arch': model_arch,
                  'hidden_units': hidden_units,
                  'learning_rate': lr,
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict()}
    
    torch.save(checkpoint, save_dir)
    
    
def predict(image, model, top_num=5):
    # Process image
    img = process_image(image)
    
    # Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)
    
    # Probs
    probs = torch.exp(model.forward(model_input))
    
    # Top probs
    top_probs, top_labs = probs.topk(top_num)
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    return top_probs, top_labels, top_flowers

