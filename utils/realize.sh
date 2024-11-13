#!/bin/bash
#SBATCH --job-name=realize_dataset
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=5:0:0
#SBATCH --mail-user=<nath0020@e.ntu.edu.sg>
#SBATCH --mail-type=ALL

mkdir -p /home/nteo/scratch/cosmo_data/

echo preparing environment...

module purge
module load python/3.10
source /home/nteo/projects/def-douglas/nteo/cosmo_project/venv/bin/activate

echo realizing dataset...

python /home/nteo/projects/def-douglas/nteo/cosmo_data/blob_realization.py

rsync -a /home/nteo/scratch/cosmo_data/ /home/nteo/projects/def-douglas/nteo/cosmo_data

echo complete

