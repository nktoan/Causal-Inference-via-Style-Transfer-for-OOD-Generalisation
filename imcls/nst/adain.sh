#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
module purge
module load Anaconda3
source activate toan_research
cd ~/toan_mammoth/
which python
# python ./utils/main.py --model scr --dataset seq-cifar10p --lr 0.1 --buffer_size 20 --batch_size 32 --minibatch_size 64 --n_epochs 50 --csv_log