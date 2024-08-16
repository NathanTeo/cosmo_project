#!/bin/bash
#SBATCH --job-name=test_model
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --time=2:0:0
#SBATCH --mail-user=<nath0020@e.ntu.edu.sg>
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:v100l:1

run=cwgan_3a

mkdir -p /home/nteo/scratch/cosmo_runs/$run
rsync -a /home/nteo/projects/def-douglas/nteo/cosmo_runs/$run/ /home/nteo/scratch/cosmo_runs/$run

echo preparing environment...

module purge
module load python/3.10
source /home/nteo/projects/def-douglas/nteo/cosmo_project/venv/bin/activate

echo run testing...

python /home/nteo/scratch/cosmo_runs/$run/config/model_params.py $run test

rsync -a /home/nteo/scratch/cosmo_runs/$run/ /home/nteo/projects/def-douglas/nteo/cosmo_runs/$run --delete

