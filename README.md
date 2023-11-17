# Causal Inference via Style Transfer for OOD Generalisation

This repository contains the codebase for our accepted paper in the Research Track of KDD'23, entitled *Causal Inference via Style Transfer for Out-of-distribution Generalisation*.

Thank you for your interest in our work!

## Structure of the Repository

The repository is structured into two distinct sub-repositories:

1. **Dassl.Pytorch**: the [Dassl.pytorch] toolbox in which we built our project on. We thank the authors of [Dassl.pytorch] for their great codebase. 
2. **imcls**: Our main repository contains our proposed Causal Inference via Style Transfer methodology for the problem of OOD Generalisation.

## How to install

This code is based on the toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). Please follow the instructions at https://github.com/KaiyangZhou/Dassl.pytorch#installation to install `dassl`.

## How to run

Please follow the steps below before running the script

- modify `DATA` and `DASSL` in `*.sh` based on the paths on your environment
- activate the `dassl` environment via `conda activate dassl`
- `cd` to `scripts/`

### Domain Generalization

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


