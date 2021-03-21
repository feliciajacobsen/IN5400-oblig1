# PyTorch stuff
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

# Built-in Python modules
import copy
import time
import os
from pathlib import Path
from typing import Callable, Optional

# Non-built-in Python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image
from sklearn.metrics import average_precision_score

# Local files
from src.vocparseclslabels import PascalVOC
from src.getimagenetclasses import *

# Seeds
np.random.seed(12)
torch.manual_seed(12)


def setbyname2(targetmodel, name, value):

    def iteratset(obj, components, value, nametail=[]):

        if not hasattr(obj, components[0]):
            return False
        elif len(components) == 1:
            if not hasattr(obj,components[0]):
                print(f"object has not the component: {components[0]}")
                print(f"nametail: {nametail}")
                exit()
            setattr(obj, components[0], value)
            return True
        else:
            nextobj = getattr(obj, components[0])
            newtail = nametail
            newtail.append(components[0])
            return iteratset(nextobj,components[1:], value, nametail=newtail)

    components = name.split(".")
    success = iteratset(targetmodel, components, value, nametail=[])
    return success


class wsconv2(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=None, eps=1e-12):
        super(wsconv2, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.eps = eps

    def forward(self, x):
        weight = self.weight
        # standard deviation
        sigma = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1)
        nc = sigma**2 + self.eps
        weight = weight / nc.expand_as(weight)
        # Comment: I'm struggling to find std along channel dimension, and to divide the weights correctly 
        # I want to divide weights by the channel-wise weight std.
        # Return the new conv2d layer, now with standardized weights
        return torch.nn.functional.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def bntoWSconverter(model, mod="A"):

    lastwasconv2 = False
    for nm, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # replace module, and get std

            lastwasconv2 = True
            usedeps = 1e-6 # use 1e-12 if you add it to a variance term, and 1e-6 if you add it to a standard deviation term
            # put in here your wsconv2, dont forget to copy convolution weight and, if exists, the convolution bias into your wsconv2

            in_channels, out_channels, kernel_size_0, kernel_size_1 = module.weight.shape

            sigma = module.weight.view(module.weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1).expand_as(weight)
            nc = sigma**2 + usedeps
            
            if kernel_size_0 == kernel_size_1: # if both kernel sizes are equal, PyTorch use a single "int" instead of a tuple
                kernel_size = kernel_size_0
            else:
                kernel_size = (kernel_size_0, kernel_size_1)

            newconv = wsconv2(in_channels, out_channels, kernel_size, bias=module.bias, eps=usedeps)

            setbyname2(model, nm, newconv)

        elif isinstance(module, nn.BatchNorm2d):

            if lastwasconv2 is False:
                print("got disconnected batchnorm??") # Previous layer should be conv2d (tracked by lastwasconv2 parameter)
                exit()

            print(f"got one {nm}")

            # Comment: here, I want to implement the two mods (A and B) for 
            # batchnorm layers, but I could not find a way to adjust the expression.
            # The documentation:
            # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
            # shows that the class can take these params: torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            # And I don't know how to adjust the expression in that class to use either mu-hat/sigma-hat OR alpha-hat/beta-hat
            if mod == "A":
                # MOD A: modify mu->mu-hat and sigma->sigma-hat, where
                # mu-hat = mu / nc 
                # sigma-hat = sigma / nc
                pass
            elif mod == "B":
                # MOD B: modify alpha->alpha-hat and beta->beta-hat, where
                # alpha-hat = alpha / nc
                # beta-hat = beta + (alpha*mu*(nc - 1))/(sqrt(sigma**2 + eps))
                pass
            else:
                raise ValueError("Mod must be either 'A' or 'B'.")

            lastwasconv2 = False
        else:
            lastwasconv2 = False


#preprocessing: https://pytorch.org/docs/master/torchvision/models.html
#transforms: https://pytorch.org/docs/master/torchvision/transforms.html
#grey images, best dealt before transform
# at first just smaller side to 224, then 224 random crop or centercrop(224)
#can do transforms yourself: PIL -> numpy -> your work -> PIL -> ToTensor()

class dataset_imagenetvalpart(Dataset):
    def __init__(self, root_dir, xmllabeldir, synsetfile, maxnum, transform=None):
        """
        Args:

            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.xmllabeldir = xmllabeldir
        self.transform = transform
        self.imgfilenames = []
        self.labels = []
        self.ending = ".JPEG"
        self.clsdict = get_classes()

        indicestosynsets, self.synsetstoindices, synsetstoclassdescr = parsesynsetwords(synsetfile)

        for root, dirs, files in os.walk(self.root_dir):
            for ct,name in enumerate(files):
                nm = os.path.join(root, name)
                #print(nm)
                if (maxnum > 0) and (ct >= maxnum):
                    break
                self.imgfilenames.append(nm)
                label, firstname = parseclasslabel(self.filenametoxml(nm), self.synsetstoindices)
                self.labels.append(label)

    def filenametoxml(self, fn):
        f = os.path.basename(fn)

        if not f.endswith(self.ending):
            print("not f.endswith(self.ending)")
            exit()

        f = f[:-len(self.ending)] + ".xml"
        f = os.path.join(self.xmllabeldir, f)

        return f

    def __len__(self):
        return len(self.imgfilenames)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.imgfilenames[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        sample = {"image": image, "label": self.labels[idx], "filename": self.imgfilenames[idx]}

        return sample


def comparetwomodeloutputs(model1, model2, dataloader, device):

    model1.eval()
    model2.eval()

    curcount = 0
    avgdiff = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):

            if (batch_idx % 100 == 0) and (batch_idx >= 100):
                print(f"at val batchindex: {batch_idx}")

            inputs = data["image"].to(device)
            outputs1 = model1(inputs)
            outputs2 = model2(inputs)

            diff = torch.mean(torch.abs((outputs1-outputs2).flatten()))

            labels = data["label"]
            print(f"diff {diff.item()}")
            avgdiff = avgdiff*(curcount/float(curcount+labels.shape[0])) + diff.item()*(labels.shape[0]/float(curcount+labels.shape[0]))

            curcount += labels.shape[0]

    return avgdiff


#routine to test that your copied model at evaluation time works as intended
def test_WSconversion(use_gpu=True, mod="A"):

    config = dict()
    config["use_gpu"] = use_gpu
    config["lr"] = 0.008 # 0.005
    config["batchsize_train"] = 2
    config["batchsize_val"] = 64

    #data augmentations
    data_transforms = {
        "val": transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            #transforms.RandomHorizontalFlip(), # we want no randomness here :)
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    root_dir = "/itf-fi-ml/shared/IN5400/dataforall/mandatory1/imagenet300/"
    xmllabeldir = "/itf-fi-ml/shared/IN5400/dataforall/mandatory1/val/"
    synsetfile = "/itf-fi-ml/shared/IN5400/dataforall/mandatory1/students/synset_words.txt"

    dset = dataset_imagenetvalpart(root_dir, xmllabeldir, synsetfile, maxnum=64, transform=data_transforms["val"])
    dataloader = torch.utils.data.DataLoader(dset, batch_size=64, shuffle=False) #, num_workers=1)

    if config["use_gpu"] is True:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = models.resnet18(pretrained=True)

    model2 = copy.deepcopy(model.to("cpu"))

    ####################
    # assumes it changes the model in-place, use model2= bntoWSconverter(model) if your routine instead modifies a copy of model and returns it
    ######################
    bntoWSconverter(model2, mod)

    model = model.to(device)
    model2 = model2.to(device)

    avgdiff = comparetwomodeloutputs(model, model2, dataloader, device)

    print(f"model checking averaged difference: {avgdiff}")  # order 1e-3 is okay, 1e-2 is still okay.


if __name__ == "__main__":
    test_WSconversion()