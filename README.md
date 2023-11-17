# Causal Inference via Style Transfer for OOD Generalisation

This repository contains the codebase for our accepted paper in the Research Track of KDD'23, entitled *Causal Inference via Style Transfer for Out-of-distribution Generalisation*.

Thank you for your interest in our work!

## Repository Structure

The repository is organized into two distinct sub-repositories:

1. **Dassl.Pytorch**: This sub-repository contains the [Dassl.pytorch] toolbox, upon which our project is built. We express our gratitude to the authors of [Dassl.pytorch] for providing an excellent codebase.
2. **imcls**: The main repository hosts our proposed methodology for Causal Inference via Style Transfer, specifically designed for addressing the Out-of-Distribution (OOD) Generalisation problem.
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

- Please go to the following GitHub page to download pre-trained AdaIN and VGG-19 models: [AdaIN](https://github.com/MAlberts99/PyTorch-AdaIN-StyleTransfer). 
- After downloading, please place the weights into the folder: `imcls/nst/vgg_checkpoints/pretrained`
- To train the Neural Style Transfer (NST) model, use the following bash command:

```bash
# PACS | Running w/ random mixing leaving out the first domain
bash dg_st_1.sh pacs resnet18
# OfficeHome | Running w/ random mixing leaving out the third domain
bash dg_st_3.sh office_home_dg resnet18 random
```
- If you do not want to pre-train the NST model, you can download and use our pre-trained for all the experimental datasets via this link: [PretrainedNST]().
- Note that the Fourier-based Style Transfer (FST) model need not be trained. 

#### Step 2: Training the classifier

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

## Citation

If you use the codes or datasets in this repository, please cite our paper.
```
@inproceedings{nguyen2023causal,
  title={Causal Inference via Style Transfer for Out-of-distribution Generalisation},
  author={Nguyen, Toan and Do, Kien and Nguyen, Duc Thanh and Duong, Bao and Nguyen, Thin},
  booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={1746--1757},
  year={2023}
}
```


