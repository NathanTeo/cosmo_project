"""
Author: Nathan Teo

Removes unwanted logs
"""

import shutil
import os 

to_remove = [
    'lightning_logs',
    'wandb'
]

try:
    os.system('sq')
    print('Please ensure no training is ongoing')
except:
    pass

max_count = 3
counter = 0

while counter<max_count:
    confirm = input('confirm (y/n):')
    if confirm=='y':
        
        for dir in to_remove:
            try:    
                shutil.rmtree(f'./{dir}')
                print(f'removed {dir}')
            except FileNotFoundError:
                print(f'directory: {dir} does not exist or already deleted')
        print('logs cleared')
                
    elif confirm=='n':
        print('aborted')
        break
    else:
        print('invalid input')
        counter+=1

if counter==max_count:
    print('aborted')
