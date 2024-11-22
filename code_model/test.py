"""
Author: Nathan Teo

This script runs testing on the model.
It is designed to only perform CPU reliant tasks, 
hence sample generation is handled in a separate script and should be performed prior to running testing.
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
    dataset.prep_data_no_gen(BlobDataModule, model_dict)
    
    # test dataset
    if generation_params['blob_size']>0:
        tester = blobTester(dataset)
        tester.test(testing_restart=testing_restart)
    elif generation_params['blob_size']==0:
        pass
    
    