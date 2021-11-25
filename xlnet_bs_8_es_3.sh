#!/bin/bash
#SBATCH --account=def-ichiro
#SBATCH --time=7-00:00:00
#SBATCH --output=run_output/xlnet_cv_output_%A_%a.out
#SBATCH --gres=gpu:v100:1
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --array=2-6
#SBATCH --mem=180G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.

module load nixpkgs/16.09
module load python/3.8.2
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip3 install --upgrade --no-binary numpy==1.20.0 numpy==1.20.0
pip install --no-index torch
pip install --no-index -r requirements.txt


echo "Starting Task"
python xlnet_cv.py --ml 512 --bs 8 --epochs 500 --lr $SLURM_ARRAY_TASK_ID --es 3
