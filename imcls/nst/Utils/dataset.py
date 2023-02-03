import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from PIL import Image
import os
import numpy as np

# Images Dataset that returns one style and one content image. As I only trained using 40.000
# images each, each image is randomly sampled. The way it is implemented does not allow multi-threading. However
# as this network is relatively small and training times low, no improved class was implemented.

class Images(Dataset): 
  def __init__(self, root_dir1, root_dir2, transform=None):
    self.root_dir1 = root_dir1
    self.root_dir2 = root_dir2
    self.transform = transform

  def __len__(self):
    return min(len(os.listdir(self.root_dir1)), len(os.listdir(self.root_dir2)))

  def __getitem__(self, idx):
    all_names1, all_names2 = os.listdir(self.root_dir1), os.listdir(self.root_dir2)
    idx1, idx2 = np.random.randint(0, len(all_names1)), np.random.randint(0, len(all_names2))

    img_name1, img_name2 = os.path.join(self.root_dir1, all_names1[idx1]), os.path.join(self.root_dir2, all_names2[idx2])
    image1 = Image.open(img_name1).convert("RGB")
    image2 = Image.open(img_name2).convert("RGB")

    if self.transform:
      image1 = self.transform(image1)
      image2 = self.transform(image2)

    coin_flip = np.random.randint(2)

    if (coin_flip == 0):
      return image1, image2
    else:
      return image2, image1  