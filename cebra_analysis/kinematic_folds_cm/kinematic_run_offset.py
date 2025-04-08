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



####### CEBRA MODEL #######



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
data_dict = data_load(input_folder, variables_to_drop, skipdates)
x = data_dict["X_pd"]
y = data_dict["y_pd"]
animals_id = data_dict["animals_id_pd"]
run_id = data_dict["run_id_pd"]
groups = [a+i for a, i in zip(animals_id, run_id)]
rng = np.random.default_rng(seed=42) 
unique_groups = np.unique(groups)  
rng.shuffle(unique_groups)   
gkf = GroupKFold(n_splits=11)
gkf.get_n_splits(groups=groups)

X_unique = np.arange(len(unique_groups))  

for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_unique, groups=unique_groups)):
        train_groups = unique_groups[train_idx]
        test_groups  = unique_groups[test_idx]
        
        train_mask = np.isin(groups, train_groups)
        test_mask  = np.isin(groups, test_groups)
        
        X_train, y_train = x.iloc[train_mask, :], y[train_mask]
        X_test,  y_test  = x.iloc[test_mask, :],  y[test_mask]
    
    
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
    
        X_train = X_train.iloc[:, :36]
        X_test = X_test.iloc[:, :36]
    
        multi_cebra_model.fit(X_train, y_train)
        tmp_file = Path("/home/garullie/CEBRA_analysis/CEBRA_train/higher_offset/models", f'run_splits_{fold_idx}_pd.pt')
        multi_cebra_model.save(tmp_file)
        
        embedding = multi_cebra_model.transform(X_train)
        test_embedding = multi_cebra_model.transform(X_test)
        
        decoder = cebra.KNNDecoder()
        decoder.fit(embedding, y_train)
        score = decoder.score(test_embedding, y_test)
        prediction = decoder.predict(test_embedding)
        y_test_ = np.array(y_test, dtype=np.int64)
        prediction_ = np.array(prediction, dtype=np.int64)
        
        labels = results = {"true": y_test_, "prediction": prediction_, "score": score}
        
        with open(f"./kinematic_predictions/predictions_kin_{fold_idx}_pd.pkl", 'wb') as handle:
                        pkl.dump(labels, handle)