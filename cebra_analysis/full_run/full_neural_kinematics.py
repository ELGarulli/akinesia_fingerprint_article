import sys
import cebra
import os
import tempfile
import torch
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib
import cebra.models
import cebra.models
import cebra.data
from cebra.models.model import _OffsetModel, ConvolutionalModelMixin
from torch import nn
from scipy import stats
from cebra import CEBRA
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.colors import rgb2hex
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
sys.path.append('/home/garullie/CEBRA_analysis/')
from dataset_load import data_load


@cebra.models.register("offset200-model") # --> add that line to register the model!
class Offset200Model(_OffsetModel, ConvolutionalModelMixin):

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        super().__init__(
            
            nn.Conv1d(num_neurons, num_units, 101),
            nn.GELU(),
            nn.Conv1d(num_units, num_units, 21),
            nn.GELU(),
            nn.Conv1d(num_units, num_units, 21),
            nn.GELU(),
            nn.Conv1d(num_units, num_units, 21),
            nn.GELU(),
            nn.Conv1d(num_units, num_units, 21),
            nn.GELU(),
            nn.Conv1d(num_units, num_output, 20),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    def get_offset(self) -> cebra.data.datatypes.Offset:
        return cebra.data.Offset(100, 100)

print(cebra.models.get_options('offset200-model'))

skipdates = ["220818", "220819"]
input_folder = "/sc-projects/sc-proj-cc15-ag-wenger-retune/data_kinematic_states_neural/"
VICON_FS = 200
variables_to_drop = ["ANIMAL_ID", "CONDITION", "RUN", "DATE", "EVENT", "ECOG_LEFT_fooof_a_knee", "ECOG_RIGHT_fooof_a_knee"]
#variables_to_drop = ["ANIMAL_ID", "CONDITION", "RUN", "DATE", "EVENT"]


def run_model(cond, slice_, slice_name):

    data_dict = data_load(input_folder, variables_to_drop, skipdates)#, dataset_name="neukin_dataset_baseline_w1500ms")

    
    X_train = data_dict[f"X_{cond}"]
    y_train = data_dict[f"y_{cond}"]
    animals_id = data_dict[f"animals_id_{cond}"]
    run_id = data_dict[f"run_id_{cond}"]
    
    multi_cebra_model = CEBRA(
        model_architecture = "offset200-model",
        batch_size = 512,
        temperature=0.1,
        learning_rate = 0.0005,
        max_iterations = 50000,
        time_offsets = 200,
        output_dimension = 3,
        device = "cuda",
        verbose = True,
        conditional="time_delta"
    )
    X_train = X_train.iloc[:, slice_]
    multi_cebra_model.fit(X_train, y_train)
    tmp_file = Path("/home/garullie/CEBRA_analysis/CEBRA_train/higher_offset/full_run/models", f'{slice_name}_full_{cond}.pt')
    multi_cebra_model.save(tmp_file)


cond = "pd"
slice_ = slice(36, None)
run_model(cond, slice_, "neural")

cond = "h"
slice_ = slice(36, None)
run_model(cond, slice_, "neural")

cond = "pd"
slice_ = slice(None, 36)
run_model(cond, slice_, "kinematics")

cond = "h"
slice_ = slice(None, 36)
run_model(cond, slice_, "kinematics")
