# PyTorch stuff
import torch
import torch.optim as optim
from torchvision import models, transforms

# Non-built-in Python libraries
import numpy as np
import matplotlib.pyplot as plt

# Local
from src import train

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

def custom_transform(image):
    """
    Will:
    * Convert PIL image to Tensor
    * CenterCrop if either old_height/old_width is LARGER than height/width
    * Pad if either old_height/old_width is SMALLER than height/width

    Return:
    * Image as tensor padded to size [3, 500, 500]
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


def create_model():
    lr = 0.1
    batchsize_tr = 16
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 20)
    model.fc.reset_parameters()

    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=batchsize_tr, shuffle=True)
    # losscriterion = torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction="mean")

    device = torch.device("cuda")

    model.to(device)

    mean_losses = train.train_epoch(
        model,
        trainloader,
        losscriterion,
        device,
        optimizer
    )
    print(mean_losses)


if __name__ == "__main__":
    root_dir = "./data/VOCdevkit/VOC2012"
    #val_data = train.dataset_voc(root_dir, "val", transform=custom_transform)
    train_val_data = train.dataset_voc(root_dir, "trainval", transform=custom_transform)
    train_data = train.dataset_voc(root_dir, "train", transform=custom_transform)
    create_model()
