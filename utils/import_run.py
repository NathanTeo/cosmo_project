"""
Author: Nathan Teo

Imports run folder from scratch.
Used when job timeout occurs and folder is not automatically transferred over.
"""

import os

run = input('run folder: ')

source = '/home/nteo/scratch/cosmo_runs'
dest = '/home/nteo/projects/def-douglas/nteo/cosmo_runs'

os.system(f'rsync -a {source}/{run}/ {dest}/{run}')