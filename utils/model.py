import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import tqdm
import numpy as np

try:
    from utils.config import DEVICE,CLASSES,NUM_CLASSES
except:
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Just using function")

#weights Example

#torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT

def create_model(num_classes,model_name,backbone="resnet50",weights=None):
    if weights.lower() == "true":
        weights= True
    else:
        weights= False
    

    if model_name.lower() == "fasterrcnn":
        # load Faster RCNN pre-trained model
        if backbone.lower() == "resnet50":
            if weights ==True:
                weights  = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT

            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights = weights )
        elif backbone.lower() == "resnet50v2":
            if weights ==True:
                weights  = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights = weights)
        elif backbone.lower() == "mobilehigh":
            if weights ==True:
                weights  = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT

            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights = weights)
        elif backbone.lower() == "mobilelow":
            if weights ==True:
                weights  = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT

            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights = weights)
        else:
            print("Backbone not Found")

        # get the number of input features 
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # define a new head for the detector with required number of classes
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    elif model_name.lower() == "ssdlite":
        print("Model : SSDLite")
        if backbone == "mobilenet":
            print("Backbone : Mobilenetv3")
            if weights ==True:
                    weights  = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT.DEFAULT

            model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights = weights)


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



