#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4

module load Anaconda3
source activate dassl

cd ~/Code_causal_neural_style_transfer/Code-CausalStyleTransfer/imcls
which python 

pwd

DATA=~/Code_causal_neural_style_transfer/Dataset
DASSL=~/Code_causal_neural_style_transfer/Code-CausalStyleTransfer/Dassl.pytorch

echo $DATA
echo $DASSL

DATASET=$1
TRAINER=StyleTransfer

if [ ${DATASET} == pacs ]; then
    D1=art_painting
    D2=cartoon
    D3=photo
    D4=sketch
elif [ ${DATASET} == office_home_dg ]; then
    D1=art
    D2=clipart
    D3=product
    D4=real_world
elif [ ${DATASET} == digits_dg ]; then
    D1=mnist
    D2=mnist_m
    D3=svhn
    D4=syn
elif [ ${DATASET} == vlcs ]; then
    D1=caltech
    D2=labelme
    D3=pascal
    D4=sun
elif [ ${DATASET} == mini_domainnet ]; then
    D1=clipart
    D2=painting
    D3=real
    D4=sketch
fi

for SEED in $(seq 21 21)
do
    for SETUP in $(seq 2 2)
    do
        if [ ${SETUP} == 1 ]; then
            S1=${D2}
            S2=${D3}
            S3=${D4}
            T=${D1}
        elif [ ${SETUP} == 2 ]; then
            S1=${D1}
            S2=${D3}
            S3=${D4}
            T=${D2}
        elif [ ${SETUP} == 3 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D4}
            T=${D3}
        elif [ ${SETUP} == 4 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D3}
            T=${D4}
        fi
        
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --source-domains ${S1} ${S2} ${S3} \
        --target-domains ${T} \
        --dataset-config-file ${DASSL}/configs/datasets/dg/${DATASET}.yaml \
        --config-file configs/trainers/style_transfer/${DATASET}.yaml \
        --output-dir output_style_transfer/${DATASET}/${T}/seed${SEED}
    done
done