import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX, TrainerStyleTransfer
from dassl.metrics import compute_accuracy
from dassl.modeling.ops import random_mixstyle, crossdomain_mixstyle

import os
import nst.Utils
from nst.Utils.networks import StyleTransferNetwork

from trainers.fourier import colorSpectrumMix

@TRAINER_REGISTRY.register()
class StyleTransfer(TrainerStyleTransfer):
    """Vanilla baseline.

    Slightly modified for mixstyle.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.learning_rate = 2e-4
        self.learning_rate_decay = 5e-5

        self.domain_test = self.cfg.DATASET.TARGET_DOMAINS[0]
        
        self.vgg_checkpoints_path = './nst/vgg_checkpoints'

        self.state_encoder = torch.load(os.path.join(self.vgg_checkpoints_path, "./pretrained/vgg_normalised.pth"))
        self.state_unisty_encoder = torch.load(os.path.join(self.vgg_checkpoints_path, "./pretrained/vgg_normalised.pth"))

        self.load_fromstate = True
        self.load_path = os.path.join(self.vgg_checkpoints_path, "pretrained/StyleTransfer Checkpoint Iter_ 120000.tar")
        self.gamma = torch.tensor([2], dtype=torch.float).to(self.device)

        del self.model
        torch.cuda.empty_cache()

        self.model_st = StyleTransferNetwork(self.cfg, self.device, 
                            self.state_encoder, self.state_unisty_encoder, self.learning_rate, self.learning_rate_decay, self.gamma, load_fromstate = self.load_fromstate, load_path = self.load_path)
            
        self.iters = self.model_st.iters

    def forward_backward(self, batch, random_batch):
        input, label = self.parse_batch_train(batch) #content
        input2, label2 = self.parse_batch_train(random_batch) #style
        
        self.model_st.adjust_learning_rate(self.model_st.optimiser, self.iters)
        self.model_st.adjust_learning_rate(self.model_st.optimiser_encoder, self.iters)

        loss_comb, content_loss, style_loss, encoder_loss = self.model_st(input2, input, label)
        
        loss = loss_comb 
        
        self.model_st.optimiser.zero_grad()

        loss.backward() #retain_graph = True
        
        self.model_st.optimiser.step()

        loss_summary = {
            'Combined Loss': loss.item() * 100,
            'Content Loss': content_loss.item() * 100,
            'Style Loss': style_loss.item() * 100,
        }

        # self.model_st.optimiser_decoder.zero_grad()
        # self.model_st.optimiser_encoder.zero_grad()
        # encoder_loss.backward()
        # self.model_st.optimiser_encoder.step()

        # loss_summary['Encoder Contrastive Loss'] = encoder_loss.item()

        self.iters += 1
        
        # if (self.batch_idx + 1) == self.num_batches:
        #     self.update_lr()

        return loss_summary

    def save_neural_transfer_state(self, encoder, decoder, optimiser, iters, run_dir):
        name = "StyleTransfer Checkpoint.tar".format(iters)
        torch.save({"Decoder" : decoder,
                    "Optimiser" : optimiser,
                    "iters": iters
                    }, os.path.join(self.output_dir, name))
        print("Saved : {} succesfully".format(name))

        name = "vgg_encoder_finetuned.pt"
        torch.save(encoder, os.path.join(self.output_dir, name))
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
