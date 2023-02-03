import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import sys
from PIL import Image

manualSeed = 999
torch.manual_seed(manualSeed)

path = "../Dataset/vgg_checkpoints"

pathStyleImages = "../Dataset/Raw images/kfold/art_painting/dog"
pathContentImages = "../Dataset/Raw images/kfold/photo/person" 

sys.path.append(path)

from Utils import networks
from Utils.dataset import Images

def test():
    # Path for the checkpoint, the vgg state_dict, image folder and device
    path_check = os.path.join(path, "StyleTransfer Checkpoint Iter_ 120000.tar")
    state_vgg = torch.load(os.path.join(path, "vgg_normalised.pth"), map_location=torch.device("cpu"))

    img_dir = "Images"

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    network = networks.StyleTransferNetwork(device, state_vgg, train=False, load_fromstate=True, load_path=path_check)

    transform = transforms.Compose([transforms.Resize(224),
                               #transforms.CenterCrop(224),
                               transforms.ToTensor()])

    toPIL = transforms.ToPILImage(mode="RGB")

    # Path to images
    style_path, content_path = os.path.join(path, img_dir, "style2_small.jpg"), os.path.join(path, img_dir, "NewYork_small.jpg")

    # Load image, convert to RGB, transform, add 0 dimension and move to device
    style = transform(Image.open(style_path).convert("RGB")).unsqueeze(0).to(device)
    content = transform(Image.open(content_path).convert("RGB")).unsqueeze(0).to(device)

    # generate image
    alpha = 1.0
    out = network(style, content, alpha).cpu()
    # convert to grid/image
    out = torchvision.utils.make_grid(out.clamp(min=-1, max=1), nrow=3, scale_each=True, normalize=True)
    # Make Pil
    img = toPIL(out)
    img

    # Save Image
    name = "Out.jpg"
    save_image(out, os.path.join(path, img_dir, name))

if __name__ == '__main__':
    print("Start Testing AdaIN:")
    test()
