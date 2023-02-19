import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time

from utils.custom_utils import (Averager,collate_fn, get_train_transform, get_valid_transform,SaveBestModel)
# try:
#     from utils.config import DEVICE,CLASSES,NUM_CLASSES
# except:
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # print("Just using function")

#weights Example

#torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT

def create_model(num_classes,model_name,backbone="resnet50",weights=None):
    

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


# # function for running training iterations
# def train(train_data_loader, model,train_loss_list,train_itr,train_loss_hist,optimizer):
#     print('Training')
#     # global train_itr
#     # global train_loss_list
    
#      # initialize tqdm progress bar
#     prog_bar = tqdm.tqdm(train_data_loader, total=len(train_data_loader))
    
#     for i, data in enumerate(prog_bar):
#         optimizer.zero_grad()
#         images, targets = data
        
#         #To Tensor
#         images = torch.tensor(np.array(images),dtype=(torch.float))

#         images = [images[0].to(DEVICE)]
#         images =  [image.reshape(3,image.shape[0],image.shape[1]) for image in images]
#         images = list(image.to(DEVICE) for image in images)
#         targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
# #         print(images[0].shape,targets)
#         loss_dict = model(images, targets)
#         losses = sum(loss for loss in loss_dict.values())
#         loss_value = losses.item()
#         train_loss_list.append(loss_value)
#         train_loss_hist.send(loss_value)
#         losses.backward()
#         optimizer.step()
#         train_itr += 1
    
#         # update the loss value beside the progress bar for each iteration
#         prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
#     return train_loss_list

# def validate(valid_data_loader, model,val_loss_list,val_itr,val_loss_hist,optimizer):
#     # global val_itr
#     # global val_loss_list
    
#     # initialize tqdm progress bar
#     prog_bar = tqdm.tqdm(valid_data_loader, total=len(valid_data_loader))
    
#     for i, data in enumerate(prog_bar):
#         images, targets = data
        
#         images = torch.tensor(np.array(images),dtype=(torch.float))

#         images = [images[0].to(DEVICE)]
#         images =  [image.reshape(3,image.shape[0],image.shape[1]) for image in images]
#         images = list(image.to(DEVICE) for image in images)
#         targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
#         with torch.no_grad():
#             loss_dict = model(images, targets)
#         losses = sum(loss for loss in loss_dict.values())
#         loss_value = losses.item()
#         val_loss_list.append(loss_value)
#         val_loss_hist.send(loss_value)
#         val_itr += 1
#         # update the loss value beside the progress bar for each iteration
#         prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
#     return val_loss_list


def load_model(path,weights,name,backbone):
    model = create_model(num_classes=2,model_name=name,backbone=backbone,weights=weights)
    checkpoint = torch.load("./myExperiment/exp2/best_model.pth",map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


class Model():


    
    def __init__(self,train_data_loader,valid_data_loader,epochs, model,train_loss_hist,val_loss_hist,optimizer,save_best=True):
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader

        self.epochs = epochs
        self.model = model
        self.optimizer = optimizer
        self.train_loss_hist = train_loss_hist
        self.val_loss_hist = val_loss_hist

        self.train_itr,self.val_itr = 1,1
        self.train_loss_list ,self.val_loss_list= [],[]
        self.SAVE_PLOTS_EPOCH = 1 # save loss plots after these many epochs
        self.SAVE_MODEL_EPOCH = 1 # save model after these many epochs
        if save_best == True:
            self.save_best_model = SaveBestModel()
            
    def save_path(self,OUT_DIR):
        self.OUT_DIR = OUT_DIR

    def train(self):
        
        # initialize tqdm progress bar
        prog_bar = tqdm.tqdm(self.train_data_loader, total=len(self.train_data_loader))
        
        for i, data in enumerate(prog_bar):
            self.optimizer.zero_grad()
            images, targets = data
            
            #To Tensor
            images = torch.tensor(np.array(images),dtype=(torch.float))

            images = [images[0].to(DEVICE)]
            # print(len(images),images[0].shape)
            images =  [image.reshape(3,image.shape[0],image.shape[1]) for image in images]
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            self.train_loss_list.append(loss_value)
            self.train_loss_hist.send(loss_value)
            losses.backward()
            self.optimizer.step()
            self.train_itr += 1
        
            # update the loss value beside the progress bar for each iteration
            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
        return self.train_loss_list


    def validate(self):
        # initialize tqdm progress bar
        prog_bar = tqdm.tqdm(self.valid_data_loader, total=len(self.valid_data_loader))
        
        for i, data in enumerate(prog_bar):
            images, targets = data
            images = torch.tensor(np.array(images),dtype=(torch.float))

            images = [images[0].to(DEVICE)]
            images =  [image.reshape(3,image.shape[0],image.shape[1]) for image in images]
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
            with torch.no_grad():
                loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            self.val_loss_list.append(loss_value)
            self.val_loss_hist.send(loss_value)
            self.val_itr += 1
            # update the loss value beside the progress bar for each iteration
            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
        return self.val_loss_list
    
    def fit(self):
        # start the training epochs
        for epoch in range(self.epochs):
            print(f"\nEPOCH {epoch+1} of {self.epochs}")
            # reset the training and validation loss histories for the current epoch
            self.train_loss_hist.reset()
            self.val_loss_hist.reset()
            # create two subplots, one for each, training and validation
            figure_1, train_ax = plt.subplots()
            figure_2, valid_ax = plt.subplots()
            # start timer and carry out training and validation
            start = time.time()
            train_loss = self.train()
            val_loss = self.validate()
            print(f"Epoch #{epoch} train loss: {self.train_loss_hist.value:.3f}")   
            print(f"Epoch #{epoch} validation loss: {self.val_loss_hist.value:.3f}")   
            end = time.time()
            print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
            
            self.save_best_model(
                self.val_loss_hist.value, epoch, self.model, self.optimizer,self.OUT_DIR
            )

            if (epoch+1) % self.SAVE_MODEL_EPOCH == 0: # save model after every n epochs
                torch.save(self.model.state_dict(), f"{self.OUT_DIR}/model_last.pth")
                print('SAVING MODEL COMPLETE...\n')

            if (epoch+1) % self.SAVE_PLOTS_EPOCH == 0: # save loss plots after n epochs
                train_ax.plot(train_loss, color='blue')
                train_ax.set_xlabel('iterations')
                train_ax.set_ylabel('train loss')
                valid_ax.plot(val_loss, color='red')
                valid_ax.set_xlabel('iterations')
                valid_ax.set_ylabel('validation loss')
                figure_1.savefig(f"{self.OUT_DIR}/train_loss.png")
                figure_2.savefig(f"{self.OUT_DIR}/valid_loss.png")
                print('SAVING PLOTS COMPLETE...')

            if (epoch+1) == self.epochs: # save loss plots and model once at the end
                train_ax.plot(train_loss, color='blue')
                train_ax.set_xlabel('iterations')
                train_ax.set_ylabel('train loss')
                valid_ax.plot(val_loss, color='red')
                valid_ax.set_xlabel('iterations')
                valid_ax.set_ylabel('validation loss')
                figure_1.savefig(f"{self.OUT_DIR}/train_loss.png")
                figure_2.savefig(f"{self.OUT_DIR}/valid_loss.png")
                torch.save(self.model.state_dict(), f"{self.OUT_DIR}/model_last.pth")

            plt.close('all')


        return (train_loss,val_loss),self.model


