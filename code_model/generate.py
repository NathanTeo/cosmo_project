"""
Author: Nathan Teo

This script generates samples with a trained model
"""

from code_model.data_modules import *
from code_model.models import model_dict
from code_model.testers.plotting_utils import *
from code_model.testers.eval_utils import *
from code_model.testers.modules import *

def run_generation(training_params, generation_params, testing_params):
    print('generate.py checkpoint reached')
    
    # prepare dataset of real and generated images
    dataset = testDataset(generation_params, training_params, testing_params)
    dataset.prep_data(BlobDataModule, model_dict, testing_restart=True)