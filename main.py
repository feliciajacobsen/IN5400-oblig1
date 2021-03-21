from src import task1, task2, gui

if __name__ == "__main__":

    # Part 1: PascalVOC dataset with ResNet18
    root_dir = "/itf-fi-ml/shared/IN5400/dataforall/mandatory1/VOCdevkit/VOC2012/"
    # task1.runstuff(root_dir, use_gpu=True, load_model=False) # Part 1

    # gui.main() # don't do this on ml6/ml7
    
    # Part 2: ImageNet dataset with modified ResNet18 
    # (conv2d layers with weight standardization, and modified batchnorm layers)
    # task2.test_WSconversion(use_gpu=True, mod="A") # Part 2
