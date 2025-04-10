import cebra
from cebra import CEBRA
import cebra.models
import os
from scipy import stats
import tempfile
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import rgb2hex
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import LocalOutlierFactor



def label_to_float(y):
    y_ = [0. if i=="fog" else 1. if i=="nlm" else 2. for i in y]
    return np.asarray(y_)

def label_to_int(y):
    y_ = [0 if i=="fog" else 1 if i=="nlm" else 2 for i in y]
    return np.asarray(y_)


def set_outlyer_to_median(x, q1=0.10, q3=0.9):
    Q1 = x.quantile(q1)
    Q3 = x.quantile(q3)
    IQR = Q3 - Q1
    median = np.median(x)
    outliers = ((x < (Q1 - 1.5 * IQR)) | (x > (Q3 + 1.5 * IQR)))
    x_clean = np.where(outliers, median, x)
    
    return x_clean

def drop_variable(df, vars_to_drop):
    xt = df.drop(vars_to_drop, axis=1)
    cols = xt.columns
    return xt, cols

def min_max_norm(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


def prep_data(df, vars_to_drop):
    ## drop unwanted variables
    xt = df.drop(vars_to_drop, axis=1)
    ## outlier rejection (set to median)
    normed_xt = xt.apply(set_outlyer_to_median, axis=0)
    ## feat normalization
    #normed_xt = normed_xt.apply(min_max_norm, axis=0)
    normed_xt = stats.zscore(normed_xt, axis=0, nan_policy="raise")
    normed_xt.dropna(axis=1, how="any", inplace=True)
    return normed_xt
    
    

def data_load(input_folder, variables_to_drop, skipdates, dataset_name="neukin_dataset_baseline", norm=True):
    
    y_train_pd = []
    x_train_pd = []
    y_train_h = []
    x_train_h = []
    animals_id_pd = []
    run_n_pd = []
    animals_id_h = []
    run_n_h = []

    days = [i for i in next(os.walk(input_folder))[1] if i not in skipdates]
    for day in days:
        path = "/".join([input_folder, day]) + "/"
        animals = next(os.walk(path))[1]
        for animal in animals:
            path = "/".join([input_folder, day, animal]) + "/"
            runways = [f.name for f in os.scandir(path) if f.is_dir()]
            for r in runways:
                try:
                    trial_path = "/".join([path, r]) + "/"
                    neukin_df = pd.read_pickle(f"{trial_path}{dataset_name}.pkl")
                    neukin_df.dropna(axis=0, how="any", inplace=True)
                    try:
                        if neukin_df["CONDITION"].values[0] == "PD":
                            x_train_pd.append(neukin_df)
                            y = np.asarray(neukin_df.loc[:, ["EVENT"]]).flatten()
                            y_train_pd.append(label_to_int(y))

                            animals_id_pd.append(neukin_df["ANIMAL_ID"])
                            run_n_pd.append(neukin_df["DATE"].astype(str)+neukin_df["RUN"].astype(str))
                        else:
                            x_train_h.append(neukin_df)
                            y = np.asarray(neukin_df.loc[:, ["EVENT"]]).flatten()
                            y_train_h.append(label_to_int(y))

                            animals_id_h.append((neukin_df["ANIMAL_ID"]))
                            run_n_h.append(neukin_df["DATE"].astype(str)+neukin_df["RUN"].astype(str))
                    except IndexError:
                        print(f"The dataset for {day} {animal} {r} was too short. Len: {len(neukin_df)}")     
                except FileNotFoundError:
                    print(f"No file found for {day} {animal} {r}")

    x_pd = pd.concat(x_train_pd)
    x_h = pd.concat(x_train_h)
    y_pd = np.concatenate(y_train_pd)
    y_h = np.concatenate(y_train_h)
    a_pd = np.concatenate(animals_id_pd)
    a_h = np.concatenate(animals_id_h)
    r_pd = np.concatenate(run_n_pd)
    r_h = np.concatenate(run_n_h)
    
    if norm:
        x_pd = prep_data(x_pd, variables_to_drop)
        x_h = prep_data(x_h, variables_to_drop)
    
    return {"X_pd": x_pd,
           "y_pd": y_pd,
           "X_h": x_h,
           "y_h": y_h,
           "animals_id_pd": a_pd,
           "animals_id_h": a_h,
           "run_id_pd": r_pd,
           "run_id_h": r_h}