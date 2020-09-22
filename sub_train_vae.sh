#!/bin/bash
#SBATCH  --output=log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G

source /scratch_net/bmicdl03/jonatank/conda/etc/profile.d/conda.sh shell.bash hook
conda activate pytorch9

python -u run_train_vae.py --model_name VAE_CamCAN_Unsupervised --config conf/conf_vae.yaml
