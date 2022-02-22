#!/bin/bash
#SBATCH --account=rpp-ichiro
#SBATCH --time=5-00:00:00
#SBATCH --output=run_output/xlnet_nr_lemma_%A.out
#SBATCH --gres=gpu:v100l:4
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=4G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.

module load nixpkgs/16.09
module load python/3.8.2
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip3 install --upgrade --no-binary numpy==1.20.0 numpy==1.20.0
pip install --no-index torch
pip install --no-index -r requirements.txt
pip install wandb --upgrade

wandb login $API_KEY

echo "Starting Task"
python xlnet_cv.py --ml 512 --bs 8 --epochs 500 --lr 5 --es 4 --lemma True
