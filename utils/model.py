import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import tqdm
import numpy as np


from utils.config import DEVICE,CLASSES,NUM_CLASSES

# import sys        
# sys.path.append('../') 
# from frcnn import train_loss_list,val_loss_list,train_loss_hist,val_loss_hist

def create_model(num_classes):
    
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model


# function for running training iterations
def train(train_data_loader, model,train_loss_list,train_itr,train_loss_hist,optimizer):
    print('Training')
    # global train_itr
    # global train_loss_list
    
     # initialize tqdm progress bar
    prog_bar = tqdm.tqdm(train_data_loader, total=len(train_data_loader))
    
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        
        #To Tensor
        images = torch.tensor(np.array(images),dtype=(torch.float))

        images = [images[0].to(DEVICE)]
        images =  [image.reshape(3,image.shape[0],image.shape[1]) for image in images]
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
#         print(images[0].shape,targets)
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)
        losses.backward()
        optimizer.step()
        train_itr += 1
    
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return train_loss_list

def validate(valid_data_loader, model,val_loss_list,val_itr,val_loss_hist,optimizer):
    print('Validating')
    # global val_itr
    # global val_loss_list
    
    # initialize tqdm progress bar
    prog_bar = tqdm.tqdm(valid_data_loader, total=len(valid_data_loader))
    
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = torch.tensor(np.array(images),dtype=(torch.float))

        images = [images[0].to(DEVICE)]
        images =  [image.reshape(3,image.shape[0],image.shape[1]) for image in images]
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)
        val_itr += 1
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_loss_list



