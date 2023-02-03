import random
from contextlib import contextmanager
import torch
import torch.nn as nn
import numpy as np


class colorSpectrumMix(nn.Module):
    def __init__(self, alpha = 1.0, ratio = 1.0):
        super().__init__()
        self.alpha = alpha
        self.ratio = ratio
    
    def forward(self, img1, img2):
        """Input image size: ndarray of [BS, C, H, W]"""
        lam = np.random.uniform(0, self.alpha)
        
        assert img1.shape == img2.shape
        c, h, w = img1.shape[1], img1.shape[2], img1.shape[3]
        h_crop = int(h * (self.ratio ** (1/2)))
        w_crop = int(w * (self.ratio ** (1/2)))
        
        # print(f'h_crop: {h_crop} and w_crop: {w_crop}') 112 - 112
        
        h_start = h // 2 - h_crop // 2
        w_start = w // 2 - w_crop // 2
        #print(f'h_start: {h_start} and w_start: {w_start}') 56 - 56
        
        img1_fft = torch.fft.fft2(img1, dim=(-2, -1))
        img2_fft = torch.fft.fft2(img2, dim=(-2, -1))
        img1_abs, img1_pha = torch.abs(img1_fft), torch.angle(img1_fft)
        img2_abs, img2_pha = torch.abs(img2_fft), torch.angle(img2_fft)
        
        img1_abs = torch.fft.fftshift(img1_abs, dim=(-2, -1))
        img2_abs = torch.fft.fftshift(img2_abs, dim=(-2, -1))        

        img1_abs_ = torch.clone(img1_abs)
        img2_abs_ = torch.clone(img2_abs)
        
        img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
            lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                                h_start:h_start + h_crop,
                                                                                                w_start:w_start + w_crop]

        img1_abs = torch.fft.ifftshift(img1_abs, dim=(-2, -1))
        img21 = img1_abs * (torch.exp(1j * img1_pha))
        img21 = torch.real(torch.fft.ifft2(img21, dim=(-2, -1)))
        
        return img21, None       
        