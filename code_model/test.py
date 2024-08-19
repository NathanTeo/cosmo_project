"""
Author: Nathan Teo

This script contains runs testing on the model:
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
    logs_plotter = logsPlotter(dataset)
    logs_plotter.plot_logs(BlobDataModule, model_dict, testing_restart)
    
    # test dataset
    if generation_params['blob_size']>0:
        tester = blobTester(dataset)
        tester.test()
    elif generation_params['blob_size']==0:
        pass    
    
    