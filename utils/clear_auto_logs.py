import shutil

to_remove = [
    'lightning_logs',
    'wandb'
]

for dir in to_remove:
    try:    
        shutil.rmtree(f'./{dir}')
        print(f'removed {dir}')
    except FileNotFoundError:
        print(f'directory: {dir} does not exist or already deleted')