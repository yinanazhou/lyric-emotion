#!/bin/bash
#SBATCH --account=def-ichiro
#SBATCH --time=20:00
#SBATCH --output=run_output/test_output_01.out
#SBATCH --gres=gpu:v100:1
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=4  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=12500M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.

module load python/3.8
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip3 install --upgrade --no-binary numpy==1.20.0 numpy==1.20.0
pip install --no-index -r requirements.txt


echo "Starting Task"
python test.py
