import glob
import os

import numpy as np
import pandas as pd

list_all_neukin_df = []

skipdates = ["220818", "220819"]
input_folder = "/sc-projects/sc-proj-cc15-ag-wenger-retune/data_kinematic_states_neural/"
VICON_FS = 200
window = 300
overlap = 250

days = [i for i in next(os.walk(input_folder))[1] if i not in skipdates]
for day in days:
    path = "/".join([input_folder, day]) + "/"
    animals = next(os.walk(path))[1]
    for animal in animals:
        path = "/".join([input_folder, day, animal]) + "/"
        runways = [f.name for f in os.scandir(path) if f.is_dir()]

        for r in runways:

            trial_path = "/".join([path, r]) + "/"

            neural_feat_file = glob.glob(trial_path + f"{r}_FEATURES.csv")[0]
            neural_df = pd.read_csv(neural_feat_file, index_col=0)

            kin_feat_file = glob.glob(trial_path + f"kin_feat_{r}_w1500ms.csv")[0]
            kin_df = pd.read_csv(kin_feat_file, index_col=0)
            frames_to_drop = int(np.floor(VICON_FS * neural_df["time"].iloc[0] / (1000*window)))
            kin_df = kin_df[frames_to_drop:]
            
            neural_df.drop(["time"], inplace=True, axis=1)
            
            delta = len(kin_df) - len(neural_df)
            if delta > 100:
                print(day, animal, r, delta)
                continue
            if delta > 0:
                kin_df = kin_df[delta+frames_to_drop:]
            if delta < 0:
                raise ValueError(f"The neural dataset is unexpectedly longer than the kin one"
                                 f" for run {day} {animal} {r}")
            kin_df.reset_index(inplace=True, drop=True)

            nueral_kinematic_dataset = pd.concat([kin_df, neural_df], axis=1)

            
            if not len(neural_df) == len(nueral_kinematic_dataset):
                print(f"len neural is: {len(neural_df)} v. len complete dataset is: {len(complete_neural_df)}")
                raise ValueError("The resulting neural kinematic dataframe is not of the correct shape")
            
            nueral_kinematic_dataset.to_pickle(f"{trial_path}neukin_dataset_baseline_w1500ms.pkl")

