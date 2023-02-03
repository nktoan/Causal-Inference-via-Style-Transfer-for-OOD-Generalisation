import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import sys

manualSeed = 42
torch.manual_seed(manualSeed)

path = "../Dataset/vgg_checkpoints"

domains = ['art_painting', 'cartoon', 'photo', 'sketch']

pathStyleImages = "../Dataset/PACS/images/art_painting/dog"
pathContentImages = "../Dataset/PACS/images/photo/person" 

sys.path.append(path)

from Utils import networks
from Utils.dataset import Images

def save_state(decoder, optimiser, iters, run_dir):
  
  name = "StyleTransfer Checkpoint Iter: {}.tar".format(iters)
  torch.save({"Decoder" : decoder,
              "Optimiser" : optimiser,
              "iters": iters
              }, os.path.join(path, name))
  print("Saved : {} succesfully".format(name))
  
def training_loop(device, all_img, network, # StyleTransferNetwork
                  dataloader_comb, # DataLoader
                  n_epochs, # Number of Epochs
                  run_dir # Directory in which the checkpoints and tensorboard files are saved
                  ):
  writer = SummaryWriter(os.path.join(path, run_dir))
  # Fixed images to compare over time
  fixed_batch_style, fixed_batch_content = all_img[0]
  fixed_batch_style, fixed_batch_content =  fixed_batch_style.unsqueeze(0).to(device), fixed_batch_content.unsqueeze(0).to(device) # Move images to device

  writer.add_image("Style", torchvision.utils.make_grid(fixed_batch_style))
  writer.add_image("Content", torchvision.utils.make_grid(fixed_batch_content))

  iters = network.iters

  for epoch in range(1, n_epochs+1):
    tqdm_object = tqdm(dataloader_comb, total=len(dataloader_comb))

    for style_imgs, content_imgs in tqdm_object:
      network.adjust_learning_rate(network.optimiser, iters)
      style_imgs = style_imgs.to(device)
      content_imgs = content_imgs.to(device)

      loss_comb, content_loss, style_loss = network(style_imgs, content_imgs)

      network.optimiser.zero_grad()
      loss_comb.backward()
      network.optimiser.step()

      # Update status bar, add Loss, add Images
      tqdm_object.set_postfix_str("Combined Loss: {:.3f}, Style Loss: {:.3f}, Content Loss: {:.3f}".format(
                                  loss_comb.item()*100, style_loss.item()*100, content_loss.item()*100))
    
      if iters % 25 == 0:
        writer.add_scalar("Combined Loss", loss_comb*1000, iters)
        writer.add_scalar("Style Loss", style_loss*1000, iters)
        writer.add_scalar("Content Loss", content_loss*1000, iters)

      if (iters+1) % 2000 == 1:
        with torch.no_grad():
          network.set_train(False)
          images = network(fixed_batch_style, fixed_batch_content)
          img_grid = torchvision.utils.make_grid(images)
          writer.add_image("Progress Iter: {}".format(iters), img_grid)
          network.set_train(True)

      # if (iters+1) % 4000 == 1:
      #     save_state(network.decoder.state_dict(), network.optimiser.state_dict(), iters, run_dir)
      #     writer.close()
      #     writer = SummaryWriter(os.path.join(path, run_dir))

      iters += 1

    #Save after each epoch
    save_state(network.decoder.state_dict(), network.optimiser.state_dict(), iters, run_dir)
    writer.close()
    writer = SummaryWriter(os.path.join(path, run_dir))
      
def train():
  transform = transforms.Compose([transforms.Resize(224), 
                              transforms.CenterCrop(224),
                              transforms.ToTensor()])

  all_img = Images(pathStyleImages, pathContentImages, transform=transform)
  device = ("cuda" if torch.cuda.is_available() else "cpu")

  learning_rate = 1e-4    
  learning_rate_decay = 5e-5

  dataloader_comb = DataLoader(all_img, batch_size=5, shuffle=True, num_workers=0, drop_last=True)
  gamma = torch.tensor([2], dtype=torch.float).to(device) # Style weight

  n_epochs = 120
  run_dir = "runs/Run 2" # Change if you want to save the checkpoints/tensorboard files in a different directory
  state_encoder = torch.load(os.path.join(path, "pretrained/vgg_normalised.pth"))
  network = networks.StyleTransferNetwork(device,
                                      state_encoder,
                                      learning_rate,
                                      learning_rate_decay,
                                      gamma,
                                      load_fromstate=False,
                                      load_path=os.path.join(path, "pretrained/StyleTransfer Checkpoint Iter_ 120000.tar"))

  training_loop(device, all_img, network, dataloader_comb, n_epochs, run_dir)

if __name__ == '__main__':
  print("Start Training AdaIN:")
  train()