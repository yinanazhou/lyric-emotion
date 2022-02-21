#!/bin/bash
#SBATCH --account=rpp-ichiro
#SBATCH --time=2:00:00
#SBATCH --output=run_output/test.out
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=8G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.

module load nixpkgs/16.09
module load python/3.8.2
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip3 install --upgrade --no-binary numpy==1.20.0 numpy==1.20.0
pip install --no-index torch
pip install --no-index -r requirements.txt
pip install --no-index wandb
pip install git+https://github.com/bonlime/pytorch-tools.git@master

wandb login $API_KEY

echo "Starting Task"
python xlnet_cv.py --ml 128 --bs 8 --epochs 100 --lr 5

