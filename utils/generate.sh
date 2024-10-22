#!/bin/bash
#SBATCH --job-name=generate_samples
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3:0:0
#SBATCH --mail-user=<nath0020@e.ntu.edu.sg>
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:v100l:1

run="$1"

echo transferring to scratch...

mkdir -p /home/nteo/scratch/cosmo_runs/$run
rsync -a /home/nteo/projects/def-douglas/nteo/cosmo_runs/$run/ /home/nteo/scratch/cosmo_runs/$run

echo preparing environment...

module purge
module load python/3.10
source /home/nteo/projects/def-douglas/nteo/cosmo_project/venv/bin/activate

echo generating samples...

python /home/nteo/scratch/cosmo_runs/$run/config/model_params.py $run generate

rsync -a /home/nteo/scratch/cosmo_runs/$run/ /home/nteo/projects/def-douglas/nteo/cosmo_runs/$run --delete

