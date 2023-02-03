import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX, TrainerXVGG
from dassl.metrics import compute_accuracy
from dassl.modeling.ops import random_mixstyle, crossdomain_mixstyle
from dassl.optim import build_optimizer, build_lr_scheduler
from torch.optim import SGD, Adam

import os
import nst.Utils
from nst.Utils.networks import StyleTransferNetwork, Encoder, Decoder

class VGG_Classifier(nn.Module):
    def __init__(self, model, pooling_layer, classifier):
        super().__init__()
        self.encoder = model
        self.classifier = classifier
        self.pooling_layer = pooling_layer

    def forward(self, x):
        x = self.classifier(torch.flatten(self.pooling_layer(self.encoder(x)), 1))
        return x

@TRAINER_REGISTRY.register()
class TrainerVGG(TrainerXVGG):
    """Vanilla baseline.

    Slightly modified for mixstyle.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.vgg_checkpoints_path = "./nst/vgg_checkpoints"

        self.state_encoder = torch.load(os.path.join(self.vgg_checkpoints_path, "pretrained/vgg_normalised.pth"))
    
        del self.model
        torch.cuda.empty_cache()

        self.encoder_vgg = Encoder(self.state_encoder, self.device)
        self.pooling_layer = nn.AdaptiveAvgPool2d((1, 1)).to(self.device) #(bs, 512)
        self.head = nn.Sequential(
            nn.Linear(512, self.num_classes),
        ).to(self.device)
        self.model = VGG_Classifier(self.encoder_vgg, self.pooling_layer, self.head)
        
        self.optim = Adam(self.model.parameters(), lr=self.cfg.OPTIM.LR)

        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model_vgg", self.model, self.optim, self.sched)

    def forward_backward(self, batch):
        input, label = self.parse_batch_train(batch)
        output =  self.model(input)
        loss = F.cross_entropy(output, label)
        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def save_encoder(self):
        name = "vgg_encoder_finetuned.pt"
        torch.save(self.model.encoder.vgg19.state_dict(), os.path.join(self.output_dir, name))
        print("Saved : {} succesfully".format(name))

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    @torch.no_grad()
    def vis(self):
        pass
