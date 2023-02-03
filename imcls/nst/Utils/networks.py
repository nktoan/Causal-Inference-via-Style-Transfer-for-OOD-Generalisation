import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from .utils import *
from .simclrloss import SupConLoss
from copy import deepcopy

# The style transfer network
class StyleTransferNetwork(nn.Module):
  def __init__(self, cfg,
                device, # "cpu" for cpu, "cuda" for gpu
                enc_state_dict, enc_state_dict_unistyle, # The state dict of the pretrained vgg19
                learning_rate=1e-4,
                learning_rate_decay=5e-5, # Decay parameter for the learning rate
                gamma=2.0, # Controls importance of StyleLoss vs ContentLoss, Loss = gamma*StyleLoss + ContentLoss
                train=True, # Wether or not network is training
                load_fromstate=False, # Load from checkpoint?
                load_path=None, # Path to load checkpoint
                scr_temperature=0.1,
                ):
    super().__init__()

    if load_fromstate and not os.path.isfile(load_path):
      raise ValueError("Checkpoint file not found")

    self.learning_rate = learning_rate
    self.learning_rate_decay = learning_rate_decay
    self.train = train
    self.gamma = gamma

    self.encoder = Encoder(enc_state_dict, device) # A pretrained vgg19 is used as the encoder (universal styles)
    self.encoder_unisty = Encoder(enc_state_dict_unistyle, device)

    self.decoder = Decoder().to(device)
    # self.do_augmentation = do_augmentation(cfg.INPUT.SIZE, cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD,
    #                                       cfg.INPUT.COLORJITTER_B, cfg.INPUT.COLORJITTER_C, cfg.INPUT.COLORJITTER_S, cfg.INPUT.COLORJITTER_H)

    # self.optimiser_encoder = optim.SGD(self.encoder.parameters(), lr=self.learning_rate)

    self.optimiser = optim.Adam(self.decoder.parameters(), lr=self.learning_rate)
    self.optimiser_encoder = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)

    self.iters = 0

    self.scr_temperature = scr_temperature
    self.pooling_layer = nn.AdaptiveAvgPool2d((1, 1)).to(device) #(bs, 512)
    self.projection_head = nn.Sequential(
      nn.Linear(512, 512),
      nn.ReLU(inplace=True),
      nn.Linear(512, 128),
    ).to(device)
    
    self.contrastive_loss = SupConLoss(temperature = self.scr_temperature)

    if load_fromstate:
      state = torch.load(load_path, map_location=torch.device("cuda"))
      print(f"Load checkpoint from {load_path}")
      self.decoder.load_state_dict(state["Decoder"])
      self.optimiser.load_state_dict(state["Optimiser"])
      self.iters = state["iters"]

  def set_train(self, boolean): # Change state of network
    assert type(boolean) == bool
    self.train = boolean

  def adjust_learning_rate(self, optimiser, iters): # Simple learning rate decay
    lr = self.learning_rate / (1.0 + self.learning_rate_decay * iters * 1.5)
    for param_group in optimiser.param_groups:
        param_group['lr'] = lr

  def forward(self, style, content, label_content = None, alpha=1.0): # Alpha can be used while testing to control the importance of the transferred style
    ## TO-DO: Contrastive Loss
    # Make the augmented batch of content

    # with torch.no_grad():
    #   aug_content = self.do_augmentation(content)

    # aug_content = aug_content.to(self.device)
    loss_encoder = None

    layers_style = self.encoder_unisty(style, self.train) # if train: returns all states
    layer_content = self.encoder(content, False) # for the content only the last layer is important

    if (label_content != None):
      # layer_aug_content = self.encoder(aug_content, False)

      # Compute the embedding vectors of content
      embed_content = torch.flatten(self.pooling_layer(layer_content), 1)
      embed_content = F.normalize(self.projection_head(embed_content), dim = 1)

      # embed_aug_content = torch.flatten(self.pooling_layer(layer_aug_content), 1)

      # combined_feats = torch.cat([embed_content.unsqueeze(1), embed_aug_content.unsqueeze(1)], dim=1)
      combined_feats = torch.unsqueeze(embed_content, 1)

      loss_encoder = self.contrastive_loss(combined_feats, label_content)
    
    # Transfer Style
    if self.train:
      style_applied = AdaIn(layer_content, layers_style[-1]) # Last layer is "style" layer
    else:
      style_applied = alpha*AdaIn(layer_content, layers_style) + (1-alpha)*layer_content # Alpha controls magnitude of style

    # Scale up
    style_applied_upscaled = self.decoder(style_applied)
    if not self.train:
      return style_applied_upscaled # When not training return transformed image

    # Compute Loss
    layers_style_applied = self.encoder(style_applied_upscaled, self.train)

    content_loss = Content_loss(layers_style_applied[-1], layer_content)
    style_loss = Style_loss(layers_style_applied, layers_style)

    loss_comb = content_loss + self.gamma*style_loss

    return loss_comb, content_loss, style_loss, loss_encoder

# The decoder is a reversed vgg19 up to ReLU 4.1. To note is that the last layer is not activated.

class Decoder(nn.Module):
  def __init__(self):
    super().__init__()

    self.padding = nn.ReflectionPad2d(padding=1) # Using reflection padding as described in vgg19
    self.UpSample = nn.Upsample(scale_factor=2, mode="nearest")

    self.conv4_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=0)

    self.conv3_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
    self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
    self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
    self.conv3_4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0)

    self.conv2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
    self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0)

    self.conv1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
    self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=0)


  def forward(self, x):
    out = self.UpSample(F.relu(self.conv4_1(self.padding(x))))

    out = F.relu(self.conv3_1(self.padding(out)))
    out = F.relu(self.conv3_2(self.padding(out)))
    out = F.relu(self.conv3_3(self.padding(out)))
    out = self.UpSample(F.relu(self.conv3_4(self.padding(out))))

    out = F.relu(self.conv2_1(self.padding(out)))
    out = self.UpSample(F.relu(self.conv2_2(self.padding(out))))

    out = F.relu(self.conv1_1(self.padding(out)))
    out = self.conv1_2(self.padding(out))
    return out

# A vgg19 Sequential which is used up to Relu 4.1. To note is that the
# first layer is a 3,3 convolution, different from a standard vgg19

class Encoder(nn.Module):
    def __init__(self, state_dict, device):
        super().__init__()
        self.vgg19 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True), # First layer from which Style Loss is calculated
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(inplace=True), # Second layer from which Style Loss is calculated
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            nn.ReflectionPad2d(padding=1), # Third layer from which Style Loss is calculated
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True), # This is Relu 4.1 The output layer of the encoder.
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True)
            ).to(device)

        self.vgg19.load_state_dict(state_dict)

        encoder_children = list(self.vgg19.children())
        self.EncoderList = nn.ModuleList([nn.Sequential(*encoder_children[:4]), # Up to Relu 1.1
                                          nn.Sequential(*encoder_children[4:11]), # Up to Relu 2.1
                                          nn.Sequential(*encoder_children[11:18]), # Up to Relu 3.1
                                          nn.Sequential(*encoder_children[18:31]), # Up to Relu 4.1, also the
                                          ])                                       # input for the decoder

    def forward(self, x, intermediates=False): # if training use intermediates = True, to get the output of
        states = []                            # all the encoder layers to calculate the style loss
        for i in range(len(self.EncoderList)):
            x = self.EncoderList[i](x)

            if intermediates:       # All intermediate states get saved in states
                states.append(x)
        if intermediates:
            return states
        return x


# from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale

# def normalize(x, mean, std):
#     assert len(x.shape) == 4
#     return (x - torch.tensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3)) \
#            / (torch.tensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3))

# class do_augmentation():
#     def __init__(self, size, mean, std, _b, _c, _s, _h):
#       self.size = size
#       self.mean = mean
#       self.std = std
#       self.transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.RandomResizedCrop(size=(size, size), scale=(0.7, 1.)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomApply([transforms.ColorJitter(
#           brightness=_b,
#           contrast=_c,
#           saturation=_s,
#           hue=_h,
#         )], p = 0.8),
#         transforms.RandomGrayscale(p=0.2),
#         transforms.Normalize(mean=self.mean, std=self.std),
#         transforms.ToTensor()
#       ])

#     def denormalize(self, x):
#       print(f'Shape of x before denormalizing: {x.shape}')
#       print(f'Before {x}')
#       for t, m, s in zip(x, self.mean, self.std):
#         t.mul_(s).add_(m)
#       print(f'After{x}')
#       print(x[0].mean(0))
#       return x

#     def __call__(self, x):
#       x = self.denormalize(x)
#       return self.transform(x)
