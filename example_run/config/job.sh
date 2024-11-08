#!/bin/bash
#SBATCH --job-name=run_name
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --time=5:0:0
#SBATCH --mail-user=<nath0020@e.ntu.edu.sg>
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:v100l:1

run=run_name

mkdir -p /home/nteo/scratch/cosmo_runs/$run
rsync -a /home/nteo/projects/def-douglas/nteo/cosmo_runs/$run/ /home/nteo/scratch/cosmo_runs/$run

module purge
module load python/3.10
source /home/nteo/projects/def-douglas/nteo/cosmo_project/venv/bin/activate

python /home/nteo/scratch/cosmo_runs/$run/config/model_params.py $run train
python /home/nteo/scratch/cosmo_runs/$run/config/model_params.py $run test

rsync -a /home/nteo/scratch/cosmo_runs/$run/ /home/nteo/projects/def-douglas/nteo/cosmo_runs/$run --delete

