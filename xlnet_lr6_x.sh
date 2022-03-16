#!/bin/bash
#SBATCH --account=rpp-ichiro
#SBATCH --time=1-00:00:00
#SBATCH --output=run_output/xl_x_lr6_x_%A_%a.out
#SBATCH --gres=gpu:v100l:4
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=4G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --array=5,10,15
#SBATCH --mail-user=yinan.a.zhou@gmail.com
#SBATCH --mail-type=ALL

module load nixpkgs/16.09
module load python/3.8.2
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
#pip3 install --upgrade --no-binary numpy==1.20.0 numpy==1.20.0
pip install --no-index -r requirements.txt
pip install --no-index numpy==1.20.0+computacanada
pip install --no-index wandb
pip install --no-index --upgrade sagemaker
pip install --no-index torch
pip install --no-index transformers==2.5.1+computecanada
pip install --no-index tensorflow_gpu==2.3.0+computecanada
pip install --no-index Keras
pip install --no-index urllib3==1.26.7+computecanada
pip install --no-index nltk
pip install --no-index scikit-learn==0.22.1
pip install --no-index tokenizers==0.5.2


wandb login $API_KEY

echo "Starting Task"
python xlnet_cv.py --ml 512 --bs 8 --epochs 500 --lr 6 --es $SLURM_ARRAY_TASK_ID


