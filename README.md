# Causal Inference via Style Transfer for Out-of-distribution Generalisation

This repo contains the code of our accepted paper at Research Track KDD'23, entitled 'Causal Inference via Style Transfer for Out-of-distribution Generalisation'

**We will clean the code and upload pretrained weights soon. Thank you for your interest in our work!**

## How to install

This code is based on [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). Please follow the instructions at https://github.com/KaiyangZhou/Dassl.pytorch#installation to install `dassl`.

## How to run

Please follow the steps below before running the script

- modify `DATA` and `DASSL` in `*.sh` based on the paths on your computer
- activate the `dassl` environment via `conda activate dassl`
- `cd` to `scripts/`

### Domain Generalization

#### Step 1: Training the neural style transfer (NST) model

- Please go the github page to download pretrained AdaIN and VGG-19 models: [AdaIN](https://github.com/MAlberts99/PyTorch-AdaIN-StyleTransfer). 
- After downloading, please place the weights into the folder: `imcls/nst/vgg_checkpoints/pretrained`
- To train the NST model, using the following bash command:

```bash
# PACS | Running w/ random mixing leaving out the first domain
bash dg_st_1.sh pacs resnet18
# OfficeHome | Running w/ random mixing leaving out the third domain
bash dg_st_3.sh office_home_dg resnet18 random
```


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


