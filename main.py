# PyTorch stuff
import torch
import torch.optim as optim
from torchvision import models, transforms

# Non-built-in Python libraries
import numpy as np
import matplotlib.pyplot as plt

# Local
from src import train


if __name__ == "__main__":
    
    use_gpu = True

 
    train.runstuff(use_gpu)
    


# TODO:
# setup:
# - write a dataloader for pascal VOC (train and val datasets)
# - create model: resnet-18 with 20 outputs (20 binary classifiers)
# - define loss: minimize 20 binary classifiers, how to do this?

# results:
# - avg precision score on validation set (one score for every of the 20 classes)
# - GUI
# - report: choose 3 RANDOM classes: show top-10 and worst-10 images for all of these 3 classes
# - calculate Tailacc(t) for all 20 classes. Repeat 10-20 times with various t
