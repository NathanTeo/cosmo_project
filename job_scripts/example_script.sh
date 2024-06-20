#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=1:0:0    
#SBATCH --mail-user=<nath0020@e.ntu.edu.sg>
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:v100l:1

cd ~/$cosgan
module purge
module load python/3.10
source ~/venv/bin/activate

python myScript.py