import pandas as pd
import py_neuromodulation as py_nm
import numpy as np
import os
import sys
import glob
from joblib import Parallel, delayed
sys.path.append('./')
sys.path.append('./')
sys.path.append('./neurokin/')
#sys.path.append('./neurokin/experiments/')
#sys.path.append('./neurokin/utils/')
#sys.path.append('./neurokin/utils/neural/')
#sys.path.append('./neurokin/utils/helper/')
#sys.path.append('./neurokin/utils/visual/')
#sys.path.append('./neurokin/utils/kinematics/')

from py_neuromodulation import nm_define_nmchannels, nm_settings

from neurokin.kinematic_data import KinematicDataRun
from neurokin.neural_data import NeuralData
from neurokin.utils.neural import processing, importing
from neurokin.utils.helper import load_config

final_fs = 0.8 #200
segment_feat_len = 1500 #333

# the order here assumes ECOGL, ECOR, LFPL, LFPR
channels_dict = {  # "NWE00052": [6, 3, , ],
    # "NWE00053": [1, 4, , ],
    # "NWE00054": [1, 4, , ],
    "NWE00089": [1, 0],
    "NWE00090": [1, 0],
    "NWE00092": [1, 0],
    "NWE00093": [1, 0],
    "NWE00130": [3, 0],
    "NWE00131": [2, 1],
    "NWE00158": [3, 0],
    "NWE00159": [3, 0],
    "NWE00166": [3, 0],
    "NWE00160": [3, 0],
    "NWE00161": [3, 0],
    "NWE00162": [3, 0],
    "NWE00163": [3, 0],
    "NWE00164": [3, 0]}
VICON_FS = 200
STREAM_NAME = "LFP1"
skipdates = ["220818", "220819"]


NUM_CHANNELS = 2
CONFIGPATH = "./config.yaml"


d = {'name': ["ECOGL", "ECOGR"],
     'used': [1, 1],
     'target': [0, 0],
     'type': ["ecog", "ecog"],
     'status': ["good", "good"],
     "new_name": ["ECOG_LEFT", "ECOG_RIGHT"],
     "rereference": ["none", "none"]}

nm_channels = pd.DataFrame(d)
settings = py_nm.nm_settings.get_default_settings()
settings = py_nm.nm_settings.reset_settings(settings)

settings["sampling_rate_features_hz"] = final_fs
settings["segment_length_features_ms"] = segment_feat_len
settings["preprocessing"] = ["raw_resampling", "re_referencing"]

settings["features"]["fft"] = True
settings["features"]["stft"] = True
settings["features"]["bandpass_filter"] = True
settings["features"]["sharpwave_analysis"] = True
settings["features"]["raw_hjorth"] = True
settings["features"]["return_raw"] = True
settings["features"]["coherence"] = True
settings["features"]["fooof"] = True
settings["features"]["bursts"] = True

settings["raw_resampling_settings"]["resample_freq_hz"] = 1000

settings["fft_settings"]["windowlength_ms"] = segment_feat_len
settings["stft_settings"]["windowlength_ms"] = int(segment_feat_len/2)

settings["bandpass_filter_settings"]["segment_lengths_ms"] = {"low beta": segment_feat_len,
                                                              "high beta": segment_feat_len,
                                                              "low gamma": int(segment_feat_len/3),
                                                              "high gamma": int(segment_feat_len/3),
                                                              "HFA": int(segment_feat_len/3)
                                                              }
settings["frequency_ranges_hz"].pop("theta")
settings["frequency_ranges_hz"].pop("alpha")
settings["burst_settings"]["frequency_bands"] = ["low beta", "high beta", "low gamma", "high gamma"]

settings["coherence"]["channels"] = [["ECOG_RIGHT", "ECOG_LEFT"]]
# ["STN_RIGHT", "STN_LEFT"],  #TODO older animals have missing channels. Discuss what to do
# ["STN_RIGHT", "ECOG_RIGHT"],
# ["STN_RIGHT", "ECOG_LEFT"],
# ["STN_LEFT", "ECOG_RIGHT"],
# ["STN_LEFT", "ECOG_LEFT"]

settings["coherence"]["frequency_bands"] = ["low beta", "high beta", "low beta", "high gamma"]
settings["fooof"]["windowlength_ms"] = int(0.8*int(segment_feat_len))  # ~ 80% of 333 since it was 800 when window was 1000
settings["fooof"]["freq_range_hz"] = [2, 120]  # TODO including high gamma range, is this right?


def get_trial_roi(run_path, configpath):
    kin_data = KinematicDataRun(run_path, configpath)
    kin_data.load_kinematics()
    kin_data.get_c3d_compliance()
    first_frame = kin_data.trial_roi_start
    last_frame = kin_data.trial_roi_end
    return first_frame, last_frame


def reorder_channels(raw_array, ch_list):
    reordered = raw_array[ch_list, :]
    return reordered


def get_neural_roi(raw, fs, first_frame, last_frame, kin_fs):
    t_onset = first_frame / kin_fs
    t_end = last_frame / kin_fs

    s_on = importing.time_to_sample(t_onset, fs=fs, is_t1=True)
    s_end = importing.time_to_sample(t_end, fs=fs, is_t2=True)

    neural_cropped_to_roi = raw[::, s_on:s_end]

    return neural_cropped_to_roi


index = os.environ["SLURM_ARRAY_TASK_ID"]
print(index)
input_folder = "/sc-projects/sc-proj-cc15-ag-wenger-retune/data_kinematic_states_neural/"
days = [i for i in next(os.walk(input_folder))[1] if i not in skipdates]
run_p = load_config.read_config("./settings_dict.yaml")[int(index)]
animal = run_p.split("/")[-3]
trial_path = f"{input_folder}{run_p}"
ch_list = channels_dict[animal]

print(trial_path)
run_path = glob.glob(trial_path + "*.c3d")[0]

first_frame, last_frame = get_trial_roi(run_path=run_path, configpath=CONFIGPATH)

neural_path = trial_path + next(os.walk(trial_path))[1][0]

neural_data = NeuralData(path=neural_path)
neural_data.load_tdt_data(stream_name=STREAM_NAME)
fs = neural_data.fs
raw = neural_data.raw
raw_roi = get_neural_roi(raw=raw, fs=fs, last_frame=last_frame, first_frame=first_frame,
                         kin_fs=VICON_FS)
raw_reordered = reorder_channels(raw_array=raw_roi, ch_list=ch_list)[:NUM_CHANNELS]
row_mean = np.mean(raw_reordered, axis=1).reshape((NUM_CHANNELS, 1))
raw_avg_normed = raw_reordered - row_mean
raw_reshaped = np.array(raw_avg_normed, dtype=np.float64)

stream = py_nm.Stream(
    settings=settings,
    nm_channels=nm_channels,
    verbose=True,
    sfreq=fs,
    line_noise=50
)
features = stream.run(raw_reshaped)
run = run_p.split("/")[-2]
run = run + "_w1500ms"
output_path = "/".join(trial_path.split("/")[:-2])
print(output_path)
stream.save_features(out_path_root=output_path, folder_name=run, feature_arr=features)
