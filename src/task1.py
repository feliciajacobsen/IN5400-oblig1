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

# Seeds
np.random.seed(12)
torch.manual_seed(12)

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
        print(f"\rTraining. Batch ({i+1}/{len(trainloader)}). Loss={np.mean(losses):1.2g}", end="")
    print(" ") # Go to new line
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

            inputs = data["image"].to(device)
            outputs = model(inputs)

            labels = data["label"] # has shape (20,) 
            loss = criterion(outputs, labels.to(device))
            losses.append(loss.item())

            concat_pred = np.concatenate((concat_pred, outputs.cpu()), axis=0)
            concat_labels = np.concatenate((concat_labels, labels), axis=0)

            for fil in data["filename"]:
                fnames.append(fil)

            print(f"\rValidation. Batch ({batch_idx+1}/{len(dataloader)}). Loss={loss.item():1.2f} ", end="")

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
        print(f"\nEpoch {epoch+1:2d}/{num_epochs:2d}")
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

        avgperfmeasure = np.mean(perfmeasure)
        print(f"\nAverage Performance Measure={avgperfmeasure:1.3f}")
        # print(f"* Classwise performance measure: ", perfmeasure)

        if avgperfmeasure > best_measure: #higher is better or lower is better?
            bestweights = model.state_dict()
            best_measure = avgperfmeasure
            best_epoch = epoch
            # print(f"current best {best_measure} at epoch {best_epoch+1}.")

    return best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs, concat_labels, concat_pred, fnames


def runstuff(root_dir, use_gpu=True, load_model=False):

    # Define configuration parameters
    config = dict()
    config["use_gpu"] = use_gpu
    config["lr"] = 0.05
    config["batchsize_train"] = 16
    config["batchsize_val"] = 64
    config["maxnumepochs"] = 20
    config["scheduler_stepsize"] = 10
    config["scheduler_factor"] = 0.5
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
    if config["use_gpu"] is True:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Model
    model = models.resnet18(pretrained=True) # Pretrained resnet18
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, config["numcl"]) # Add new fully connected layer
    model.fc.reset_parameters()
    if load_model is True:
        model.load_state_dict(torch.load("./models/pretrained"))
    model = model.to(device)

    losscriterion = torch.nn.BCEWithLogitsLoss(reduction="mean")#torch.nn.BCEWithLogitsLoss(reduction="mean")

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.fc.parameters(), lr=config["lr"], momentum=0.9) #optim.RMSprop(model.fc.parameters(), lr=config["lr"]) 

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

    model_dir = "./models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_dir + "/pretrained")

    # Plots and results
    fig_dir = "./figures"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # train/test loss vs. epochs
    epoch_arr = np.arange(1, config["maxnumepochs"] + 1)
    plt.figure()
    plt.plot(epoch_arr, trainlosses, label="Train Loss")
    plt.plot(epoch_arr, testlosses, label="Test Loss")
    plt.xlabel("Epoch number")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.savefig("./figures/traintestloss_vs_epochs.jpg")

    # aPM vs. epochs
    fig = plt.figure()
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height]) # shrink width
    testperfsarr = np.array(testperfs)
    for i, c in enumerate(class_names):
        ax.plot(epoch_arr, testperfsarr[:,i], label=c) 
    plt.xlabel("Epoch number")
    plt.ylabel("Average Performance Measure")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # position label to the right of plot
    plt.grid()
    plt.savefig("./figures/test_apm.jpg")

    # Saving top10/bottom10 images for all classes
    for c in class_names:
        directory = "./figures/" + c
        top_dir = directory + "/top"
        bottom_dir = directory + "/bottom"
        if not os.path.exists(directory):
            os.makedirs(directory)
        if not os.path.exists(top_dir):
            os.makedirs(top_dir)
        if not os.path.exists(bottom_dir):
            os.makedirs(bottom_dir)

    image_path = Path(root_dir) / "JPEGImages" 
    N_classes = 20 # Pick N random classes
    top_bottom_N = 10 # 
    for c in np.random.permutation(20)[:N_classes]: # Loop over N classes
        cl_name = class_names[c]
        # Obtain sorted indices of the prediction scores of ONE class
        sorted_indices = np.argsort(concat_pred[:,c], axis=0)
        # Sort all predicition scores for ONE class from lowest to highest
        sorted_preds = concat_pred[sorted_indices, c]
        # Sort ground truth labels accordingly
        sorted_labels = concat_labels[sorted_indices, c]

        # Collect sorted scores ONLY for when this class is present (avoid ground truth=absence)
        sorted_indices_where_ground_truth_present = []

        for label, idx in zip(sorted_labels, sorted_indices):
            if label != 0:
                sorted_indices_where_ground_truth_present.append(int(idx)) # Sorted indices (but zero-labels are disregarded)
        
        for j in range(top_bottom_N):

            top_j = sorted_indices_where_ground_truth_present[-(j+1)]
            top_image = PIL.Image.open(image_path / (fnames[top_j] + ".jpg"), "r").convert("RGB")
            top_image.save(f"./figures/{cl_name}/top/top_{j+1}.jpg")

            bottom_j = sorted_indices_where_ground_truth_present[j]
            bottom_image = PIL.Image.open(image_path / (fnames[bottom_j] + ".jpg"), "r").convert("RGB")
            bottom_image.save(f"./figures/{cl_name}/bottom/bottom_{j+1}.jpg")

    # tail accuracy analysis
    def tailacc(t, preds, labels):
        tail = np.zeros(t.shape)
        # compute tailacc for each t
        for i, t_val in enumerate(t):
            I_soft = np.logical_and(preds > t_val, labels == 1)
            I_hard = np.logical_and(preds > 0.5, labels == 1)
            tail[i] = np.sum(I_soft*I_hard) / (np.sum(I_soft) + 1e-16)

        return tail

    t = np.linspace(0.5,np.max(concat_pred),20) # from hard limit (0.5 for sigmoid) to largest prediction score
    tail = np.zeros(t.shape)
    for c in range(config["numcl"]):
        tail += tailacc(t, concat_pred[:,c], concat_labels[:,c])
    tail /= config["numcl"]

    # tailacc plot
    plt.figure()
    plt.title("Average Tailacc(t) of all classes in PascalVOC")
    plt.plot(t, tail)
    plt.grid()
    plt.xlabel("t")
    plt.ylabel("tailacc(t)")
    plt.savefig("./figures/tailacc.jpg")


if __name__ == "__main__":
    runstuff()
