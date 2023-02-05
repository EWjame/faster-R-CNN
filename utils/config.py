import yaml
import torch
import argparse

#Device for train images
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Create the parser
parser = argparse.ArgumentParser()

# Add an argument
# parser.add_argument('--name', type=str, required=True)
parser.add_argument('-bs','--batchsize', type=int,default=4)
parser.add_argument('-img','--imgsize', type=tuple,default=(512,512))
parser.add_argument('-e','--epoch',type=int,default=2)

# Parse the argument
args = parser.parse_args()



#load data path
with open("./data/data.yaml", "r") as stream:
    try:
        data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


CLASSES = data['class']
NUM_CLASSES = len(CLASSES)
