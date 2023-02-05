import torch
import yaml

import os
import cv2
import tqdm as tqdm
import numpy as np
import glob as glob
import time
import matplotlib.pyplot as plt
# from config import CLASSES, RESIZE_TO, TRAIN_DIR, VALID_DIR, BATCH_SIZE
from torch.utils.data import Dataset, DataLoader
# from utils import collate_fn, get_train_transform, get_valid_transform

#import utils file
from utils.config import data,DEVICE,args,CLASSES
from utils.load_img import LoadDataset
from utils.model import train,validate,create_model
from utils.custom_utils import (Averager,collate_fn, get_train_transform, get_valid_transform,SaveBestModel)





BATCH_SIZE = args.batchsize # increase / decrease according to GPU memeory
RESIZE_TO = args.imgsize # resize the image for training and transforms
NUM_EPOCHS = args.epoch # number of epochs to train for
# training images and XML files directory
# TRAIN_DIR = './data/pascalVoc_oral/train'
TRAIN_DIR = '.'+ data['train']

# validation images and XML files directory
# VALID_DIR = './data/pascalVoc_oral/valid'
VALID_DIR = '.'+ data['valid']
# classes: 0 index is reserved for background
# CLASSES = ['background', 'oral']


print(BATCH_SIZE,RESIZE_TO,TRAIN_DIR)


NUM_CLASSES = len(CLASSES)
# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False
# location to save model and plots
OUT_DIR = './outputs'
if os.path.isdir(OUT_DIR) == False:
    os.mkdir(OUT_DIR)
index = 1
while os.path.isdir(OUT_DIR+f"/output{index}") == True:
    index+=1
OUT_DIR = OUT_DIR+f"/output{index}"
print(OUT_DIR)
os.mkdir(OUT_DIR)

SAVE_PLOTS_EPOCH = 2 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2 # save model after these many epochs


train_dataset = LoadDataset(TRAIN_DIR, RESIZE_TO[0], RESIZE_TO[1], CLASSES)
valid_dataset = LoadDataset(VALID_DIR, RESIZE_TO[0], RESIZE_TO[1], CLASSES)

print(train_dataset.all_images[0])
print(train_dataset.dir_path)

#Data Loader

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}\n")



# initialize the Averager class
train_loss_hist = Averager()
val_loss_hist = Averager()
train_itr = 1
val_itr = 1
# train and validation loss lists to store loss values of all...
# ... iterations till ena and plot graphs for all iterations
train_loss_list = []
val_loss_list = []

model = create_model(num_classes=NUM_CLASSES)
model = model.to(DEVICE)
# get the model parameters
params = [p for p in model.parameters() if p.requires_grad]
# define the optimizer
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

# name to save the trained model with
MODEL_NAME = 'model'
# whether to show transformed images from data loader or not
if VISUALIZE_TRANSFORMED_IMAGES:
    from utils import show_tranformed_image
    show_tranformed_image(train_loader)

# initialize SaveBestModel class
save_best_model = SaveBestModel()

# start the training epochs
for epoch in range(NUM_EPOCHS):
    print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
    # reset the training and validation loss histories for the current epoch
    train_loss_hist.reset()
    val_loss_hist.reset()
    # create two subplots, one for each, training and validation
    figure_1, train_ax = plt.subplots()
    figure_2, valid_ax = plt.subplots()
    # start timer and carry out training and validation
    start = time.time()
    train_loss = train(train_loader, model,train_loss_list,train_itr,train_loss_hist,optimizer)
    val_loss = validate(valid_loader, model,val_loss_list,val_itr,val_loss_hist,optimizer)
    print(f"Epoch #{epoch} train loss: {train_loss_hist.value:.3f}")   
    print(f"Epoch #{epoch} validation loss: {val_loss_hist.value:.3f}")   
    end = time.time()
    print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
    
    save_best_model(
        val_loss_hist.value, epoch, model, optimizer,OUT_DIR
    )

    if (epoch+1) % SAVE_MODEL_EPOCH == 0: # save model after every n epochs
        torch.save(model.state_dict(), f"{OUT_DIR}/model{index}.pth")
        print('SAVING MODEL COMPLETE...\n')

    if (epoch+1) % SAVE_PLOTS_EPOCH == 0: # save loss plots after n epochs
        train_ax.plot(train_loss, color='blue')
        train_ax.set_xlabel('iterations')
        train_ax.set_ylabel('train loss')
        valid_ax.plot(val_loss, color='red')
        valid_ax.set_xlabel('iterations')
        valid_ax.set_ylabel('validation loss')
        figure_1.savefig(f"{OUT_DIR}/train_loss_{index}.png")
        figure_2.savefig(f"{OUT_DIR}/valid_loss_{index}.png")
        print('SAVING PLOTS COMPLETE...')

    if (epoch+1) == NUM_EPOCHS: # save loss plots and model once at the end
        train_ax.plot(train_loss, color='blue')
        train_ax.set_xlabel('iterations')
        train_ax.set_ylabel('train loss')
        valid_ax.plot(val_loss, color='red')
        valid_ax.set_xlabel('iterations')
        valid_ax.set_ylabel('validation loss')
        figure_1.savefig(f"{OUT_DIR}/train_loss_{index}.png")
        figure_2.savefig(f"{OUT_DIR}/valid_loss_{index}.png")
        torch.save(model.state_dict(), f"{OUT_DIR}/model{index}.pth")

    plt.close('all')