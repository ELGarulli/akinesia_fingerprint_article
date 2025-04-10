import pandas as pd
import py_neuromodulation as py_nm
import numpy as np
import os
import sys
import re
import glob
from joblib import Parallel, delayed
sys.path.append('./')
sys.path.append('./')
sys.path.append('./neurokin/')
sys.path.append('./neurokin/experiments/')
#sys.path.append('./neurokin/utils/')
#sys.path.append('./neurokin/utils/neural/')
#sys.path.append('./neurokin/utils/helper/')
sys.path.append('./neurokin/utils/kinematics/')
from neurokin.kinematic_data import KinematicDataRun
from neurokin.experiments.neural_correlates import (get_events_dict,
                                                    time_to_frame_in_roi)
from neurokin.utils.helper import load_config

def shift_df(df, bodyparts, axis):
    df_shifted = df.copy()
    for bp in bodyparts:
        df_shifted["scorer", bp, axis] = df_shifted["scorer", bp, axis] + abs(min(df_shifted["scorer", bp, axis]))
    return df_shifted

def window_max_occurance(input_events):
    values, counts = np.unique(input_events, return_counts=True)
    ind = np.argmax(counts)
    v_max = values[ind]
    return v_max

skipdates = ["220818", "220819"]
index = os.environ["SLURM_ARRAY_TASK_ID"]
print(index)
input_folder = "/sc-projects/sc-proj-cc15-ag-wenger-retune/data_kinematic_states_neural/"
days = [i for i in next(os.walk(input_folder))[1] if i not in skipdates]
run_p = load_config.read_config("./settings_dict.yaml")[int(index)]
animal = run_p.split("/")[-3]
run = run_p.split("/")[-2]
day = run_p.split("/")[-4]
trial_path = f"{input_folder}{run_p}"


CONFIGPATH = "./config.yaml"
TO_SHIFT = ['lankle','lmtp','lcrest','lknee','rhip','lhip','rmtp','lshoulder','rcrest','rankle','rshoulder','rknee']

pda = ["NWE00130", "NWE00160", "NWE00161", "NWE00162", "NWE00163", "NWE00164"]

window = 300
overlap = 250
skiprows = 2
vicon_fs = 200

run_path = glob.glob(trial_path + "*.c3d")[0]
kin_data = KinematicDataRun(run_path, CONFIGPATH)
kin_data.load_kinematics()
kin_data.get_c3d_compliance()
bodyparts_to_drop = [i[1] for i in kin_data.markers_df.columns.to_list()[::3] if i[1].startswith("*")]
kin_data.markers_df = kin_data.markers_df.drop(bodyparts_to_drop, axis=1, level=1, inplace=False)
kin_data.markers_df = shift_df(kin_data.markers_df, TO_SHIFT, "y")
kin_data.bodyparts = [bp for bp in kin_data.bodyparts if bp not in bodyparts_to_drop]

kin_data.extract_features()

bins_stats = kin_data.get_binned_features(window=window, overlap=overlap)
l_step_height = kin_data.get_trace_height(marker="lmtp", axis="z", window=window, overlap=overlap)
l_step_length = kin_data.get_step_fwd_movement_on_bins(marker="lmtp", axis="y", window=window,
                                                       overlap=overlap)

r_step_height = kin_data.get_trace_height(marker="rmtp", axis="z", window=window, overlap=overlap)
r_step_length = kin_data.get_step_fwd_movement_on_bins(marker="rmtp", axis="y", window=window,
                                                       overlap=overlap)

binned_features = pd.concat((bins_stats, l_step_height, l_step_length, r_step_height, r_step_length),
                            axis=1)

binned_features = binned_features  # [2:]
binned_features["ANIMAL_ID"] = animal
binned_features["CONDITION"] = "PD" if animal in pda else "H"
binned_features["RUN"] = run
binned_features["DATE"] = day

event_path = [trial_path + fname for fname in os.listdir(trial_path) if
                                              re.match(r"(?i)[a-z_-]+[0-9]{1,3}.csv", fname) and not fname.startswith("kin")][0] 
events = np.full(len(kin_data.markers_df), "nlm")

events_dict = get_events_dict(event_path=event_path,
                              skiprows=skiprows,
                              framerate=vicon_fs)

for event_type, event_value in events_dict.items():
    if "gait" in event_type:
            for i, j in event_value:
                i = time_to_frame_in_roi(timestamp=i, fs=vicon_fs, first_frame=kin_data.trial_roi_start)
                j = time_to_frame_in_roi(timestamp=j, fs=vicon_fs, first_frame=kin_data.trial_roi_start)
                events[i:j] = "gait"
    if "fog" in event_type:
            for i, j in event_value:
                i = time_to_frame_in_roi(timestamp=i, fs=vicon_fs, first_frame=kin_data.trial_roi_start)
                j = time_to_frame_in_roi(timestamp=j, fs=vicon_fs, first_frame=kin_data.trial_roi_start)
                events[i:j] = "fog"

ev_cat = pd.Series(events, dtype="category")
events_window = ev_cat.cat.codes.rolling(window=window, step=overlap).apply(window_max_occurance, raw=True)
binned_features["EVENT"] = events_window

df_slimmer = binned_features[binned_features.columns.drop(list(binned_features.filter(regex='_min')))]
df_slimmer = df_slimmer[df_slimmer.columns.drop(list(df_slimmer.filter(regex='_max')))]
df_slimmer = df_slimmer[df_slimmer.columns.drop(list(df_slimmer.filter(regex='_std')))]

df_slimmer.to_csv(f"{trial_path}/kin_feat_{run}_w1500ms.csv")