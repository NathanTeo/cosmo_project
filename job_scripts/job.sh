#!/bin/bash
run=CWGAN_run_2b

cp -r home/nteo/projects/def-douglas/nteo/cosmo_runs/$run home/nteo/scratch/cosmo_runs/

sbatch home/nteo/scratch/cosmo_runs/$run/scripts/job_params.py

module purge
module load python/3.10
source home/nteo/projects/def-douglas/nteo/cosmo_project/venv/bin/activate

python home/nteo/scratch/cosmo_runs/$run/config/model_params.py

cp -r home/nteo/scratch/cosmo_runs/$run home/nteo/projects/def-douglas/nteo/cosmo_runs 