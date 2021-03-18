# PyTorch stuff
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

# Built-in Python modules
import time
import os
from pathlib import Path
from typing import Callable, Optional

# Non-built-in Python libraries
import PIL.Image
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local files
from src.vocparseclslabels import PascalVOC


class dataset_voc(Dataset):
    def __init__(self, root_dir, trvaltest, transform=None):
        # Directory management using pathlib module
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "JPEGImages"
        self.main_dir = self.root_dir / "ImageSets" / "Main"
        # Custom transform function (to transform images)
        self.transform = transform
        class_names = PascalVOC(self.root_dir).list_image_sets()

        if trvaltest not in ["train","trainval","val","test"]:
            raise ValueError(
                "Parameter trvaltest needs to be train, trainval, val or test."
            )

        # Read trvaltest filenames (e.g. 2007_00008 etc.)
        trvaltest_filename = str(trvaltest) + ".txt"
        trvaltest_path = self.main_dir / trvaltest_filename
        file = open(trvaltest_path, mode="r")
        self.imgfilenames = file.read().split("\n")[:-1]
        self.imgpaths = [
            self.image_dir / (img + ".jpg") for img in self.imgfilenames
        ]
        file.close()

        # Save labels as a matrix
        self.labels = np.zeros((len(self.imgfilenames), len(class_names)))
        for col, cls in enumerate(class_names):
            # Define full path of class file (e.g. aeroplane_train.txt)
            cls_filename = cls + "_" + trvaltest + ".txt"
            cls_path = self.main_dir / cls_filename

            # Read file
            file = open(cls_path, mode="r")
            lines = file.readlines()

            for row, line in enumerate(lines):
                # Avoid empty lines
                if len(line) == 0:
                    continue

                img, lab = line.split(" ", 1) # can handle multi-spacing
                if int(lab) == 0: # difficult = presence of class, hence 1
                    lab = 1.0
                elif int(lab) == -1: # absence of class, hence 0
                    lab = 0.0
                self.labels[row, col] = float(lab)

            file.close()

    def __len__(self):
        return len(self.imgfilenames)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.imgpaths[idx], "r").convert("RGB")

        if self.transform:
            image = self.transform(image)

        sample = {
            "image": image,
            "label": self.labels[idx],
            "filename": self.imgfilenames[idx]
        }

        return sample


def custom_transform(image):
    """
    Will:
    * Convert PIL image to Tensor
    * CenterCrop if either old_height/old_width is LARGER than height/width
    * Pad if either old_height/old_width is SMALLER than height/width

    Return:
    * Image as tensor padded/cropped to size [3, 500, 500]
    """
    # Output dimensions
    width = 500
    height = 500

    # Convert PIL image to tensor, because the rest of the function
    # expects tensor
    tensor = transforms.ToTensor()(image)
    # tensor = transforms.Lambda(lambda image: torch.from_numpy(numpy.array(image).astype(numpy.float32)).unsqueeze(0))

    old_channels, old_height, old_width = tensor.shape

    # Crop if old_width/old_height is too large
    if (old_width > width or old_height > height):
        tensor = transforms.CenterCrop([width, height])(tensor)

    # Perform padding to increase (height*width) to (max_h*max_w)
    wpad = (width - old_width) / 2
    hpad = (height - old_height) / 2
    # Evaluate padding for left, right, top, bottom
    left = wpad if (wpad % 1 == 0 or wpad == 0) else wpad + 0.5
    right = wpad if (wpad % 1 == 0 or wpad == 0) else wpad - 0.5
    top = hpad if (hpad % 1 == 0 or hpad == 0) else hpad + 0.5
    bottom = hpad if (hpad % 1 == 0 or hpad == 0) else hpad - 0.5
    # Perform padding
    pad_fn = transforms.Pad((int(left), int(top), int(right), int(bottom)))
    tensor = pad_fn(tensor)

    # Double check that image tensor now has desired shape
    if (
        tensor.shape[0] != 3 or
        tensor.shape[1] != width or
        tensor.shape[2] != height
    ):
        msg = (
            f"custom_transform() did not successfully create desired "
            f"tensor of shape of [3, {width}, {height}], but rather created a "
            f"tensor of shape {tensor.shape}."
        )
        raise ValueError(msg)

    return tensor


def train_epoch(model, trainloader, criterion, device, optimizer):
    """
    Perform one epoch (train by using all data exactly once).
    """
    model.train()

    losses = []
    for i, sample in enumerate(trainloader):

        inputs = sample["image"].to(device)
        labels = sample["label"].to(device)

        optimizer.zero_grad() # Zero out gradients

        output = model(inputs)
        # Computing loss from output and ground truth labels
        loss = criterion(output, labels)

        loss.backward() # Computes gradients
        optimizer.step() # Update values

        losses.append(loss.item())

        if (i % 100 == 0):
            print(f"Current mean of losses={np.mean(losses):1.2g}")

    return np.mean(losses)


def mean_avg_precision(model, dataloader, criterion, device, numcl):
    model.eval()
    concat_pred = np.zeros((0, numcl))
    concat_labels = np.zeros((0, numcl))
    avgprecs = np.zeros(numcl) # Average precision for each class
    fnames = [] # Filenames as they come out of the dataloader

    with torch.no_grad():
        losses = []

        for batch_idx, data in enumerate(dataloader):

            if (batch_idx % 100 == 0) and (batch_idx >= 100):
                print(f"at val batchindex: {batch_idx}")

            inputs = data["image"].to(device)
            outputs = model(inputs)

            labels = data["label"] # has shape (20,) 

            loss = criterion(outputs, labels.to(device))
            losses.append(loss.item())

            concat_pred = np.concatenate((concat_pred, outputs.cpu()), axis=0)
            concat_labels = np.concatenate((concat_labels, labels), axis=0)

            for fil in data["filename"]:
                fnames.append(fil)
            
    for i in range(numcl):
        avgprecs[i] = average_precision_score(concat_labels[:,i], concat_pred[:,i])

    return avgprecs, np.mean(losses), concat_labels, concat_pred, fnames


def traineval2_model_nocv(dataloader_train, dataloader_test, model, criterion, optimizer, scheduler, num_epochs, device, numcl):

    best_measure = 0
    best_epoch = -1

    trainlosses = []
    testlosses = []
    testperfs = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 11)
        avgloss = train_epoch(model, dataloader_train, criterion, device, optimizer)
        trainlosses.append(avgloss)

        if scheduler is not None:
            scheduler.step()

        perfmeasure, testloss,concat_labels, concat_pred, fnames = mean_avg_precision(
            model, dataloader_test, criterion, device, numcl
        )
        testlosses.append(testloss)
        testperfs.append(perfmeasure)

        print(f"at epoch: {epoch+1}. classwise perfmeasure: ", perfmeasure)
        avgperfmeasure = np.mean(perfmeasure)
        print(f"at epoch: {epoch+1}. avgperfmeasure {avgperfmeasure}.")

        if avgperfmeasure > best_measure: #higher is better or lower is better?
            bestweights = model.state_dict()
            best_measure = avgperfmeasure
            best_epoch = epoch
            print(f"current best {best_measure} at epoch {best_epoch}.")

    return best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs, concat_labels, concat_pred, fnames



def runstuff(use_gpu=False):

    config = dict()
    root_dir = "/itf-fi-ml/shared/IN5400/dataforall/mandatory1/VOCdevkit/VOC2012/"
    # Define configuration parameters
    config["use_gpu"] = use_gpu
    config["lr"] = 0.005
    config["batchsize_train"] = 16
    config["batchsize_val"] = 64
    config["maxnumepochs"] = 1
    config["scheduler_stepsize"] = 10
    config["scheduler_factor"] = 0.3
    # kind of a dataset property
    config["numcl"] = 20

    class_names = PascalVOC(root_dir).list_image_sets()

    # data augmentations
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Datasets
    image_datasets={}
    image_datasets["train"] = dataset_voc(
        root_dir=root_dir,
        trvaltest="train",
        transform=data_transforms["train"]
    )
    image_datasets["val"] = dataset_voc(
        root_dir=root_dir,
        trvaltest="val",
        transform=data_transforms["val"]
    )

    # Dataloaders
    dataloaders = {}
    dataloaders["train"] = torch.utils.data.DataLoader(
        image_datasets["train"],
        batch_size=config["batchsize_train"],
        shuffle=True,
        num_workers=1
    )
    dataloaders["val"] = torch.utils.data.DataLoader(
        image_datasets["val"],
        batch_size=config["batchsize_val"],
        shuffle=True,
        num_workers=1
    )

    # Device
    if config["use_gpu"]:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")


    model = models.resnet18(pretrained=True) # Pretrained resnet18
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, config["numcl"]) # Add new fully connected layer
    model.fc.reset_parameters()
    model = model.to(device)

    losscriterion = torch.nn.BCEWithLogitsLoss(reduction="mean")


    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.fc.parameters(), lr=config["lr"], momentum=0.9)

    # Decay LR by a factor of 0.3 every X epochs
    somelr_scheduler = lr_scheduler.StepLR(
        optimizer, 
        step_size=config["scheduler_stepsize"], 
        gamma=config["scheduler_factor"]
    )
    

    best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs, concat_labels, concat_pred, fnames = traineval2_model_nocv(
        dataloaders["train"],
        dataloaders["val"],
        model,
        losscriterion,
        optimizer,
        somelr_scheduler,
        num_epochs = config["maxnumepochs"],
        device = device,
        numcl = config["numcl"]
    )

    # Predicitions where there are presence of object (hence multiply)
    sorted_scores = np.argsort(concat_pred * concat_labels, axis=0)

    
    N = 3 # for 3 classes
    for random_class in np.random.permutation(20)[:N]:
        top = int(sorted_scores[-1,random_class])
        # worst = int(sorted_scores[0,random_class])

        best_image_path = Path(root_dir) / "JPEGImages" / (fnames[top] + ".jpg")
        best_image = PIL.Image.open(best_image_path, "r").convert("RGB")
        plt.imshow(best_image)
        plt.title(f"Best predicted image for \n{class_names[random_class]}, prediction={concat_pred[top,random_class]:1.3}")
        plt.savefig(f"best_{class_names[random_class]}.jpg")

    
    


###########
# for part2
###########

# def setbyname2(targetmodel,name,value):
#
#     def iteratset(obj,components,value,nametail=[]):
#
#       if not hasattr(obj,components[0]):
#         return False
#       elif len(components)==1:
#         if not hasattr(obj,components[0]):
#           print("object has not the component:",components[0])
#           print("nametail:",nametail)
#           exit()
#         setattr(obj,components[0],value)
#         #print("found!!", components[0])
#         #exit()
#         return True
#       else:
#         nextobj=getattr(obj,components[0])
#
#         newtail = nametail
#         newtail.append(components[0])
#         #print("components ",components, nametail, newtail)
#         #print(type(obj),type(nextobj))
#
#         return iteratset(nextobj,components[1:],value, nametail= newtail)
#
#     components=name.split(".")
#     success=iteratset(targetmodel,components,value, nametail=[])
#     return success
#
#
#
# class wsconv2(nn.Conv2d):
#   def __init__(self, in_channels, out_channels, kernel_size, stride,
#                      padding, dilation = 1 , groups =1 , bias = None, eps=1e-12 ):
#     super(wsconv2, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
#
#     self.eps=eps
#
#   def forward(self,x):
#     #torch.nn.functional.conv2d documentation tells about weight shapes
#     pass
#
#
# def bntoWSconverter(model):
#
#   #either you modify model in place
#   #or you create a copy of it e.g. using copy.deepcopy(...)
#   # https://discuss.pytorch.org/t/are-there-any-recommended-methods-to-clone-a-model/483/17
#
#   lastwasconv2= False
#   for nm,module in model.named_modules():
#     #print(nm)
#
#     if isinstance(module, nn.Conv2d):
#       #replace, get std
#       lastwasconv2= True
#
#       usedeps= 1e-12 # use 1e-12 if you add it to a variance term, and 1e-6 if you add it to a standard deviation term
#
#       #TODO
#       # put in here your wsconv2, dont forget to copy convolution weight and, if exists, the convolution bias into your wsconv2
#
#       setbyname2(model,nm,newconv)
#
#     elif isinstance(module,nn.BatchNorm2d):
#
#       if False == lastwasconv2:
#         print("got disconnected batchnorm??")
#         exit()
#
#
#       print("got one", nm)
#
#       #TODO
#       # you will need here data computed from the preceding nn.Conv2d instance which came along your way
#
#       #delete
#       lastwasconv2= False
#
#     else:
#       lastwasconv2= False
#
#
#
#
# #preprocessing: https://pytorch.org/docs/master/torchvision/models.html
# #transforms: https://pytorch.org/docs/master/torchvision/transforms.html
# #grey images, best dealt before transform
# # at first just smaller side to 224, then 224 random crop or centercrop(224)
# #can do transforms yourself: PIL -> numpy -> your work -> PIL -> ToTensor()
#
# class dataset_imagenetvalpart(Dataset):
#   def __init__(self, root_dir, xmllabeldir, synsetfile, maxnum, transform=None):
#
#     """
#     Args:
#
#         root_dir (string): Directory with all the images.
#         transform (callable, optional): Optional transform to be applied
#             on a sample.
#     """
#
#     self.root_dir = root_dir
#     self.xmllabeldir=xmllabeldir
#     self.transform = transform
#     self.imgfilenames=[]
#     self.labels=[]
#     self.ending=".JPEG"
#
#     self.clsdict=get_classes()
#
#
#     indicestosynsets,self.synsetstoindices,synsetstoclassdescr=parsesynsetwords(synsetfile)
#
#
#     for root, dirs, files in os.walk(self.root_dir):
#        for ct,name in enumerate(files):
#           nm=os.path.join(root, name)
#           #print(nm)
#           if (maxnum >0) and ct>= (maxnum):
#             break
#           self.imgfilenames.append(nm)
#           label,firstname=parseclasslabel(self.filenametoxml(nm) ,self.synsetstoindices)
#           self.labels.append(label)
#
#
#   def filenametoxml(self,fn):
#     f=os.path.basename(fn)
#
#     if not f.endswith(self.ending):
#       print("not f.endswith(self.ending)")
#       exit()
#
#     f=f[:-len(self.ending)]+".xml"
#     f=os.path.join(self.xmllabeldir,f)
#
#     return f
#
#
#   def __len__(self):
#       return len(self.imgfilenames)
#
#   def __getitem__(self, idx):
#     image = PIL.Image.open(self.imgfilenames[idx]).convert("RGB")
#
#     label=self.labels[idx]
#
#     if self.transform:
#       image = self.transform(image)
#
#     #print(image.size())
#
#     sample = {"image": image, "label": label, "filename": self.imgfilenames[idx]}
#
#     return sample
#
#
#
#
# def comparetwomodeloutputs(model1, model2, dataloader, device):
#
#     model1.eval()
#     model2.eval()
#
#     curcount = 0
#     avgdiff = 0
#
#     with torch.no_grad():
#       for batch_idx, data in enumerate(dataloader):
#
#
#           if (batch_idx%100==0) and (batch_idx>=100):
#               print("at val batchindex: ", batch_idx)
#
#           inputs = data["image"].to(device)
#           outputs1 = model1(inputs)
#           outputs2 = model2(inputs)
#
#           diff=torch.mean(torch.abs((outputs1-outputs2).flatten()))
#
#           labels = data["label"]
#           print("diff",diff.item())
#           avgdiff = avgdiff*( curcount/ float(curcount+labels.shape[0]) ) + diff.item()* ( labels.shape[0]/ float(curcount+labels.shape[0]) )
#
#
#           curcount+= labels.shape[0]
#
#     return avgdiff
#
#
# #routine to test that your copied model at evaluation time works as intended
# def test_WSconversion():
#
#
#   config = dict()
#
#   #config["use_gpu"] = True
#   #config["lr"]=0.008 #0.005
#   #config["batchsize_train"] = 2
#   #config["batchsize_val"] = 64
#
#   #data augmentations
#   data_transforms = {
#       "val": transforms.Compose([
#           transforms.Resize(224),
#           transforms.CenterCrop(224),
#           #transforms.RandomHorizontalFlip(), # we want no randomness here :)
#           transforms.ToTensor(),
#           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#       ]),
#   }
#
#   root_dir="/itf-fi-ml/shared/IN5400/dataforall/mandatory1/imagenet300/"
#   xmllabeldir="/itf-fi-ml/shared/IN5400/dataforall/mandatory1/val/"
#   synsetfile="/itf-fi-ml/shared/IN5400/dataforall/mandatory1/students/synset_words.txt"
#
#   dset= dataset_imagenetvalpart(root_dir, xmllabeldir, synsetfile, maxnum=64, transform=data_transforms["val"])
#   dataloader =  torch.utils.data.DataLoader(dset, batch_size=64, shuffle=False) #, num_workers=1)
#
#   import copy
#   device=torch.device("cpu")
#   #model
#   model = models.resnet18(pretrained=True)
#   model2 = copy.deepcopy(model.to("cpu"))
#
#   ####################
#   # assumes it changes the model in-place, use model2= bntoWSconverter(model) if your routine instead modifies a copy of model and returns it
#   ######################
#   bntoWSconverter(model2)
#
#   model = model.to(device)
#   model2 = model2.to(device)
#
#   avgdiff = comparetwomodeloutputs(model, model2, dataloader, device)
#
#   print("model checking averaged difference", avgdiff )  # order 1e-3 is okay, 1e-2 is still okay.


if __name__=="__main__":
    runstuff()
