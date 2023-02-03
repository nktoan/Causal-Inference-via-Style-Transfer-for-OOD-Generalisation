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

@TRAINER_REGISTRY.register()
class CausalFDTest(CausalTrainer):
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

        self.style_transfer_checkpoints_path = os.path.join(os.getcwd(), f"output_neural_pretrained_style_transfer_augs/{self.learned_dataset}/StyleTransfer/style_transfer_vgg19/{self.domain_test}/seed1/StyleTransfer Checkpoint.tar")

        self.vgg_checkpoints_path = './nst/vgg_checkpoints'
        self.state_unisty_encoder = torch.load(os.path.join(self.vgg_checkpoints_path, "./pretrained/vgg_normalised.pth"))

        self.state_encoder = self.state_unisty_encoder #torch.load(self.vgg_trained_checkpoints_path)

        print(f"Load neural style transfer models from {self.style_transfer_checkpoints_path}")

        self.model_st = StyleTransferNetwork(cfg, self.device, self.state_encoder, self.state_unisty_encoder, train = False, load_fromstate = self.load_fromstate, load_path = self.style_transfer_checkpoints_path)

        self.alpha = cfg.TRAINER.CAUSALFD.ALPHA # for the content strength
        self.beta = cfg.TRAINER.CAUSALFD.BETA # for the main image

        self.normalization_transform = Normalize(mean = self.cfg.INPUT.PIXEL_MEAN, std = self.cfg.INPUT.PIXEL_STD)

        print(f"Initialize the causal front-door models with alpha = {self.alpha} and beta = {self.beta}")

    def model_inference_causal_front_door(self, input): #TESTING
        # print("Inference using front-door adjustment:")
        #take a random batch
        
        # random_ids = np.random.randint(max(0, len(self.random_train_loader_x) - 1))
        # random_batch = next(itertools.islice(self.random_train_loader_x, random_ids, None))

        self.random_train_loader_x_iteration = iter(self.random_train_loader_x)
        self.random_train_loader_x_iteration2 = iter(self.random_train_loader_x_2)
        self.random_train_loader_x_iteration3 = iter(self.random_train_loader_x_3)

        random_batch = next(self.random_train_loader_x_iteration)
        random_batch2 = next(self.random_train_loader_x_iteration2)
        random_batch3 = next(self.random_train_loader_x_iteration3)

        #parse that random batch
        input2, _, _ = self.parse_batch_train(random_batch)
        input3, _, _ = self.parse_batch_train(random_batch2)
        input4, _, _ = self.parse_batch_train(random_batch3)

        input2 = input2[torch.randperm(input.shape[0])]
        input3 = input3[torch.randperm(input.shape[0])]
        input4 = input4[torch.randperm(input.shape[0])]

        #generative counterfactual images: (z, x') -> (x'')
        with torch.no_grad():
            int_batch1 = self.model_st(input2, input, alpha = self.alpha)
            int_batch2 = self.model_st(input3, input, alpha = self.alpha)
            int_batch3 = self.model_st(input4, input, alpha = self.alpha)

        output_1 = self.model(int_batch1)
        output_2 = self.model(int_batch2)
        output_3 = self.model(int_batch3)

        #take the raw prediction
        output_raw = self.model(input)

        output_average = (output_raw * self.beta + output_1 * ((1 - self.beta) / 3) + output_2 * ((1 - self.beta) / 3)
                            + output_3 * ((1 - self.beta) / 3))

        return output_average

    def forward_backward(self, batch, random_batch, random_batch2, random_batch3): #TRAINING
        input, label, index = self.parse_batch_train(batch)

        # TO-DO: Take the raw prediction
        output_raw = self.model(input)

        # To-DO: X' sampled from P(X) - Done!
        input2, _, index2 = self.parse_batch_train(random_batch)
        input3, _, index3 = self.parse_batch_train(random_batch2)
        input4, _, index4 = self.parse_batch_train(random_batch3)

        # TO-DO: Generate counterfactual images: (z, x') -> (x'') - Done!
        with torch.no_grad():
            int_batch2 = self.model_st(input2, input, label, alpha = self.alpha)

        output_2 = self.model(int_batch2)

        output_average = output_raw * self.beta + output_2 * ((1 - self.beta) / 3)

        del output_raw, int_batch2, output_2, input2
        torch.cuda.empty_cache()

        with torch.no_grad():
            int_batch2 = self.model_st(input3, input, label, alpha = self.alpha)
            
        output_2 = self.model(int_batch2)

        output_average = output_average + output_2 * ((1 - self.beta) / 3)

        del int_batch2, output_2, input3
        torch.cuda.empty_cache()

        with torch.no_grad():
            int_batch2 = self.model_st(input4, input, label, alpha = self.alpha)
            
        output_2 = self.model(int_batch2)

        output_average = output_average + output_2 * ((1 - self.beta) / 3)

        del int_batch2, output_2, input4, _
        torch.cuda.empty_cache()

        # TO-DO: Learn on counterfactual images x'' - Done!

        # To-DO: Average the predictions
        loss = F.cross_entropy(output_average, label)

        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(output_average, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        index = batch['index']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, index
    
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
