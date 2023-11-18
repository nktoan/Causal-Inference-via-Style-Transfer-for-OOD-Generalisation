# Causal Inference via Style Transfer for OOD Generalisation

This repository contains the codebase for our accepted paper in the Research Track of **KDD'23**, titled **['Causal Inference via Style Transfer for Out-of-distribution Generalisation'](https://dl.acm.org/doi/10.1145/3580305.3599270)**.

Thank you for your interest in our work!

## Repository Structure

The repository is organised into two distinct sub-repositories:

1. **Dassl.Pytorch**: This sub-repository contains the [Dassl.pytorch] toolbox, upon which our project is built. We express our gratitude to the authors of [Dassl.pytorch] for providing an excellent codebase.
2. **imcls**: The main repository hosts our proposed methodology for Causal Inference via Style Transfer, specifically designed to address the Out-of-Distribution (OOD) Generalisation problem.
Installation Instructions

This code relies on the [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) toolbox. Please follow the instructions outlined in Dassl.pytorch's installation guide to install the `dassl` package.

Feel free to reach out if you encounter any issues during the installation process.

## Running Instructions
Before executing the script, please ensure the following steps are completed:

- Modify the paths for `DATA` and `DASSL` in the `*.sh` files according to the directory structure in your environment.
- Activate the `dassl` environment using the command `conda activate dassl`.
- Navigate to the `scripts/`.

After completing these steps, you can proceed with running the script below. If you encounter any issues or have questions, feel free to ask for assistance.

### Domain Generalisation

#### Step 1: Training the neural style transfer (NST) model

- To obtain the pre-trained AdaIN and VGG-19 models, visit the GitHub page: [AdaIN](https://github.com/MAlberts99/PyTorch-AdaIN-StyleTransfer).
(Alternatively, we have backed up the pre-trained AdaIN NST models and the VGG-19-normalised files [here](https://drive.google.com/drive/folders/1Fd0j4_7CxC_vhUFCkQUviE_2drsL84R4?usp=sharing). Utilise these resources for fine-tuning on our Out-of-Distribution (OOD) generalisation datasets.)

- After completing the download, please move the obtained weights to the designated folder: `imcls/nst/vgg_checkpoints/pretrained`
- To initiate the training/fine-tuning process for the Neural Style Transfer (NST) model, employ the following bash command:

```bash
# PACS | Running w/ random mixing leaving out the first domain
bash dg_st_1.sh pacs resnet18
# OfficeHome | Running w/ random mixing leaving out the third domain
bash dg_st_3.sh office_home_dg resnet18 random
```
- If you do not wish to pre-train the NST model, you can download and use our pre-trained model for all the experimental datasets via this link: [PretrainedNST](https://drive.google.com/drive/folders/124eDQlk04VC0jsQNCzMe016px5f9hcbM?usp=sharing).
- It's important to note that the Fourier-based Style Transfer (FST) model does not require training.
- Additionally, for Digits-DG, we resize all images to the dimensions of 224x224 before downscaling them to (32×32) for further processing by the classifier.
  
#### Step 2: Training the classifier

Please use the following command line for training the generalisable classifier:

```bash
# PACS | Running w/ random mixing leaving out the first domain
bash dg_fd_1.sh pacs resnet18 random

# PACS | Running w/ crossdomain mixing leaving out the second domain
bash dg_fd_2.sh pacs resnet18 crossdomain

# OfficeHome | Running w/ random mixing leaving out the third domain
bash dg_fd_3.sh office_home_dg resnet18 random

# OfficeHome | Running w/ crossdomain mixing leaving out the fourth domain
bash dg_fd_4.sh office_home_dg resnet18 crossdomain
```

Please be aware that *'random mixing'* involves randomly sampling stylised images from all domains, where the number of images in each domain may vary. On the other hand, *'cross-domain'* ensures the random selection of the same number of images from all domains.

If you have any questions, feel free to reach out by raising an issue on this GitHub repository or contacting me via the email provided in the paper.

## Citation

If you employ the codes or datasets provided in this repository or utilise our proposed method as comparison baselines in your experiments, please cite our paper. Again, thank you for your interest!
```
@inproceedings{nguyen2023causal,
  title={Causal Inference via Style Transfer for Out-of-distribution Generalisation},
  author={Nguyen, Toan and Do, Kien and Nguyen, Duc Thanh and Duong, Bao and Nguyen, Thin},
  booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={1746--1757},
  year={2023}
}
```


