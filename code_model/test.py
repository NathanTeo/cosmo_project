"""
Author: Nathan Teo

This script runs testing on the model:
"""

from code_model.data_modules import *
from code_model.models import model_dict
from code_model.testers.plotting_utils import *
from code_model.testers.eval_utils import *
from code_model.testers.modules import *

def run_testing(training_params, generation_params, testing_params, testing_restart=False):
    print('test.py checkpoint reached')
    
    # prepare dataset of real and generated images
    dataset = testDataset(generation_params, training_params, testing_params)
    dataset.prep_data(BlobDataModule, model_dict, testing_restart)
    
    # plot losses
    if 'GAN' in training_params['model_version']:
        logs_plotter = ganLogsPlotter(dataset)
        logs_plotter.plot_logs(BlobDataModule, model_dict, testing_restart)
    elif 'Diffusion' in training_params['model_version']:
        logs_plotter = diffLogsPlotter(dataset)
        logs_plotter.plot_logs(testing_restart)
    
    # test dataset
    if generation_params['blob_size']>0:
        tester = blobTester(dataset)
        tester.test()
    elif generation_params['blob_size']==0:
        pass    
    
    