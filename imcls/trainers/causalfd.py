import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX, TrainerStyleTransfer, CausalTrainer
from dassl.metrics import compute_accuracy
from dassl.modeling.ops import random_mixstyle, crossdomain_mixstyle

import os
import nst.Utils
from nst.Utils.networks import StyleTransferNetwork

import numpy as np
import itertools

from torchvision.transforms import Normalize

from torchvision.models import resnet18

from torchvision.transforms.functional import normalize, resize, to_pil_image

import pandas as pd
from trainers.fourier import colorSpectrumMix

@TRAINER_REGISTRY.register()
class CausalFD(CausalTrainer):
    """Causal Front-door Model For Domain Generalization (DG)"""

    def __init__(self, cfg):
        super().__init__(cfg)

        #TO-DO: load the pretrained model
        #Done!

        self.domain_test = self.cfg.DATASET.TARGET_DOMAINS[0]
        self.load_fromstate = True

        if (self.cfg.DATASET.NAME == "OfficeHomeDG"):
            self.learned_dataset = "office_home_dg"
        elif (self.cfg.DATASET.NAME == "PACS"):
            self.learned_dataset = "pacs"
        elif (self.cfg.DATASET.NAME == "DigitsDG"):
            self.learned_dataset = "digits_dg"
        elif (self.cfg.DATASET.NAME == "VLCS"):
            self.learned_dataset = "vlcs"
        elif (self.cfg.DATASET.NAME == "miniDomainNet"):
            self.learned_dataset = "mini_domainnet"

        self.style_transfer_checkpoints_path = os.path.join(os.getcwd(), f"output_style_transfer/{self.learned_dataset}/{self.domain_test}/seed21/StyleTransfer Checkpoint.tar")
        
        self.vgg_checkpoints_path = './nst/vgg_checkpoints'
        self.state_unisty_encoder = torch.load(os.path.join(self.vgg_checkpoints_path, "./pretrained/vgg_normalised.pth"))

        self.state_encoder = self.state_unisty_encoder #torch.load(self.vgg_trained_checkpoints_path)

        print(f"Load neural style transfer models from {self.style_transfer_checkpoints_path}")

        self.model_st = StyleTransferNetwork(cfg, self.device, self.state_encoder, self.state_unisty_encoder, train = False, load_fromstate = self.load_fromstate, load_path = self.style_transfer_checkpoints_path)

        self.alpha = cfg.TRAINER.CAUSALFD.ALPHA # for the content strength
        self.beta = cfg.TRAINER.CAUSALFD.BETA # for the main image
        self.weight_for_balance_NST_and_Fourier = cfg.TRAINER.CAUSALFD.BALANCED_WEIGHT #this is for NST first

        self.normalization_transform = Normalize(mean = self.cfg.INPUT.PIXEL_MEAN, std = self.cfg.INPUT.PIXEL_STD)

        print(f"Initialize the causal front-door models with alpha = {self.alpha} and beta = {self.beta}")

        self.fourier_augmentations = colorSpectrumMix(alpha = 1.0)
        
    def model_inference_causal_front_door(self, input, label = None, impath = None, tracking_files = None): #TESTING

        # print("Inference using front-door adjustment:")
        #take a random batch
        
        # random_ids = np.random.randint(max(0, len(self.random_train_loader_x) - 1))
        # random_batch = next(itertools.islice(self.random_train_loader_x, random_ids, None))

        self.random_train_loader_x_list_iteration = [iter(train_loader) for train_loader in self.random_train_loader_x_list]
        
        random_batch1, random_batch2, random_batch3 = next(self.random_train_loader_x_list_iteration[0]),\
            next(self.random_train_loader_x_list_iteration[1]), next(self.random_train_loader_x_list_iteration[2])
     
        #parse that random batch
        input2, _, domain2, _, _ = self.parse_batch_train(random_batch1)
        input3, _, domain3, _, _ = self.parse_batch_train(random_batch2)
        input4, _, domain4, _, _ = self.parse_batch_train(random_batch3)
        
        input2, input3, input4 = input2[torch.randperm(input.shape[0])], input3[torch.randperm(input.shape[0])], input4[torch.randperm(input.shape[0])]
        
        #generative counterfactual images: (z, x') -> (x'')
        with torch.no_grad():
            int_batch2 = self.model_st(input2, input, alpha = self.alpha)
            int_batch3 = self.model_st(input3, input, alpha = self.alpha)
            int_batch4 = self.model_st(input4, input, alpha = self.alpha)
            
        # input2, input3, input4 = input2[torch.randperm(input2.shape[0])], input3[torch.randperm(input3.shape[0])], input4[torch.randperm(input4.shape[0])]

        #by fourier transforms
        input21, _ = self.fourier_augmentations(input, input2) #ij: j is the original label
        input31, _ = self.fourier_augmentations(input, input3)
        input41, _ = self.fourier_augmentations(input, input4)
        
        #resize data to (32, 32) if possibly
        if (self.learned_dataset == "digits_dg"):
            reduced_size = 32
        
        if (self.learned_dataset == "digits_dg"):
            int_batch2 = F.interpolate(int_batch2, size = [reduced_size, reduced_size], mode = 'bilinear')
            int_batch3 = F.interpolate(int_batch3, size = [reduced_size, reduced_size], mode = 'bilinear')
            int_batch4 = F.interpolate(int_batch4, size = [reduced_size, reduced_size], mode = 'bilinear')
            input = F.interpolate(input, size = [reduced_size, reduced_size], mode = 'bilinear')
            input31, input21 = F.interpolate(input31, size = [reduced_size, reduced_size], mode = 'bilinear'), \
                                F.interpolate(input21, size = [reduced_size, reduced_size], mode = 'bilinear')
            input41 = F.interpolate(input41, size = [reduced_size, reduced_size], mode = 'bilinear')
            

        output_2 = self.model(self.normalization_transform(int_batch2))
        output_3 = self.model(self.normalization_transform(int_batch3))
        output_4 = self.model(self.normalization_transform(int_batch4))
        
        output_2f = self.model(self.normalization_transform(input21))
        output_3f = self.model(self.normalization_transform(input31))
        output_4f = self.model(self.normalization_transform(input41))

        #take the raw prediction
        output_raw = self.model(self.normalization_transform(input))
        
        output_average = output_raw * self.beta +\
            self.weight_for_balance_NST_and_Fourier * ((1 - self.beta) / 3) * (output_2 + output_3 + output_4) +\
            (1 - self.weight_for_balance_NST_and_Fourier) * ((1 - self.beta) / 3) * (output_2f + output_3f + output_4f)

        return output_average, tracking_files

    def forward_backward(self, batch, random_batch, random_batch2, random_batch3 = None): #TRAINING
        input, label, domain, index, _ = self.parse_batch_train(batch)
        
        # print(f'Domains in each batch: {domain}')

        # TO-DO: Take the raw prediction

        # To-DO: X' sampled from P(X) - Done!
        input2, _, domain2, index2, _ = self.parse_batch_train(random_batch)
        input3, _, domain3, index3, _ = self.parse_batch_train(random_batch2)
        input4, _, domain4, index4, _ = self.parse_batch_train(random_batch3)
        
        # TO-DO: Generate counterfactual images by Fourier: (z, x') -> (x'') 
        input21, _ = self.fourier_augmentations(input, input2) #ij: j is the original label
        input31, _ = self.fourier_augmentations(input, input3) #ij: j is the original label
        input41, _ = self.fourier_augmentations(input, input4) #ij: j is the original label
        
        # input2, input3, input4 = input2[torch.randperm(input2.shape[0])], input3[torch.randperm(input3.shape[0])], input4[torch.randperm(input4.shape[0])]
    
        # TO-DO: Generate counterfactual images by NST: (z, x') -> (x'') - Done!
        with torch.no_grad():
            int_batch2 = self.model_st(input2, input, label, alpha = self.alpha)
            int_batch3 = self.model_st(input3, input, label, alpha = self.alpha)
            int_batch4 = self.model_st(input4, input, label, alpha = self.alpha)
        
        #resize data to (32, 32) if possibly
        if (self.learned_dataset == "digits_dg"):
            reduced_size = 32
        
        if (self.learned_dataset == "digits_dg"):
            int_batch2 = F.interpolate(int_batch2, size = [reduced_size, reduced_size], mode = 'bilinear')
            int_batch3 = F.interpolate(int_batch3, size = [reduced_size, reduced_size], mode = 'bilinear')
            int_batch4 = F.interpolate(int_batch4, size = [reduced_size, reduced_size], mode = 'bilinear')
            input = F.interpolate(input, size = [reduced_size, reduced_size], mode = 'bilinear')
            input31, input21 = F.interpolate(input31, size = [reduced_size, reduced_size], mode = 'bilinear'), \
                                F.interpolate(input21, size = [reduced_size, reduced_size], mode = 'bilinear')
            input41 = F.interpolate(input41, size = [reduced_size, reduced_size], mode = 'bilinear')
            
        output_raw = self.model(self.normalization_transform(input))

        output_2 = self.model(self.normalization_transform(int_batch2))
        output_3 = self.model(self.normalization_transform(int_batch3))
        output_4 = self.model(self.normalization_transform(int_batch4))
        
        output_2f = self.model(self.normalization_transform(input21))
        output_3f = self.model(self.normalization_transform(input31))
        output_4f = self.model(self.normalization_transform(input41))
        
        output_average = output_raw * self.beta +\
            self.weight_for_balance_NST_and_Fourier * ((1 - self.beta) / 3) * (output_2 + output_3 + output_4) +\
            (1 - self.weight_for_balance_NST_and_Fourier) * ((1 - self.beta) / 3) * (output_2f + output_3f + output_4f)

        # TO-DO: Learn on counterfactual images x'' - Done!

        # To-DO: Average the predictions
        loss = F.cross_entropy(output_average, label)

        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(output_average, label)[0].item()
        }

        del int_batch2, int_batch3, output_2, output_3, input2, input3, _
        torch.cuda.empty_cache()
        
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        index = batch['index']
        impath = batch['impath']
        domain = batch['domain']
        
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, domain, index, impath

    @torch.no_grad()
    def vis(self):
        self.set_model_mode('eval')
        output_dir = self.cfg.OUTPUT_DIR
        source_domains = self.cfg.DATASET.SOURCE_DOMAINS
        print('Source domains:', source_domains)

        out_embed = []
        out_domain = []
        out_label = []

        split = self.cfg.TEST.SPLIT
        data_loader = self.val_loader if split == 'val' else self.test_loader

        print('Extracting style features')

        for batch_idx, batch in enumerate(data_loader):
            input = batch['img'].to(self.device)
            label = batch['label']
            domain = batch['domain']
            impath = batch['impath']

            # model should directly output features or style statistics
            raise NotImplementedError
            output = self.model(input)
            output = output.cpu().numpy()
            out_embed.append(output)
            out_domain.append(domain.numpy())
            out_label.append(label.numpy()) # CLASS LABEL

            print('processed batch-{}'.format(batch_idx + 1))

        out_embed = np.concatenate(out_embed, axis=0)
        out_domain = np.concatenate(out_domain, axis=0)
        out_label = np.concatenate(out_label, axis=0)
        print('shape of feature matrix:', out_embed.shape)
        out = {
            'embed': out_embed,
            'domain': out_domain,
            'dnames': source_domains,
            'label': out_label
        }
        out_path = osp.join(output_dir, 'embed.pt')
        torch.save(out, out_path)
        print('File saved to "{}"'.format(out_path))




# B = input.shape[0]
# perm = torch.arange(B - 1, -1, -1)  # inverse index

# perm_b, perm_a = perm.chunk(2)

# perm_b = perm_b[torch.randperm(perm_b.shape[0])]
# perm_a = perm_a[torch.randperm(perm_a.shape[0])]
# # perm = torch.cat([perm_b, perm_a], 0)

# if (domain2[0] == domain[0]):
#     input2, domain2, index2 = input2[torch.cat([perm_b, perm_a], 0)],\
#                                 domain2[torch.cat([perm_b, perm_a], 0)], index2[torch.cat([perm_b, perm_a], 0)]

# if (domain3[0] == domain[0]):
#     perm_b = perm_b[torch.randperm(perm_b.shape[0])]
#     perm_a = perm_a[torch.randperm(perm_a.shape[0])]
#     input3, domain3, index3 = input3[torch.cat([perm_b, perm_a], 0)],\
#                                 domain3[torch.cat([perm_b, perm_a], 0)], index3[torch.cat([perm_b, perm_a], 0)]