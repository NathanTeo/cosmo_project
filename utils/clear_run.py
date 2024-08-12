"""
Author: Nathan Teo

Clear run history. Alternative to training_restart=True. 
"""

import os

run = input('run folder: ')

max_count = 3
counter = 0

while counter<max_count:
    confirm = input('confirm (y/n):')
    if confirm=='y':
        root = f'/home/nteo/projects/def-douglas/nteo/cosmo_runs/{run}'

        os.system(f'rm -r {root}/checkpoints {root}/logs {root}/plots')

        print("history cleared")
        break
    elif confirm=='n':
        print('aborted')
        break
    else:
        print('invalid input')
        counter+=1

if counter==max_count:
    print('aborted')