import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

import pytorch_lightning as pl
import numpy as np
import pandas as pd

from modules.prediction_intervals import generate_df_mimic, generate_df_unknown_prop, intervals_known_prop

import warnings


np.random.seed(2024) 
pl.seed_everything(2024)
warnings.filterwarnings('ignore') 


# Intervals synthetic dataset with kown propensities
seed_list = [2024,0,1,2,3,4,5,6,7,8]
dataset_list = [1,2]
delta_list = [1,5,10]
covariates_list = ["X1", "X2", "X3", "X4"]
evaluation_alphas = [0.05, 0.1, 0.2]
path = '../results/intervals_simulated_known_seeds.csv' 

intervals_known_prop(seed_list, dataset_list, delta_list, covariates_list, evaluation_alphas, path)


# Intervals synthetic dataset with unkown propensities
dataset_list = [1,2]
intervention_list = [5, 7, 10]
evaluation_alphas = [0.05, 0.1, 0.2]
path = '../results/prediction_intervals_simulated_unknown_new.csv'

generate_df_unknown_prop(dataset_list, intervention_list, evaluation_alphas, path)



# Intervals on MIMIC

evaluation_alphas = [0.1]
df = pd.read_csv('../data/MIMIC/mimic_data_test.csv').dropna() 

patient_1 = df.loc[(df["gender"] == 1) & (df["age"] < 0)][0]
patient_2 = df.loc[(df["gender"] == 1) & (df["age"] > 0)][0]
patient_3 = df.loc[(df["gender"] == 0) & (df["age"] < 0)][0]
patient_4 = df.loc[(df["gender"] == 0) & (df["age"] > 0)][0]
patient_list = [patient_1, patient_2, patient_3, patient_4]


for sample in patient_list:
    sample = sample.drop(columns="A")
    sample_df = pd.DataFrame(np.repeat(sample.values, 11, axis=0))
    sample_df.columns = sample.columns
    sample_df["A"] = np.arange(0,1.1,0.1)

    generate_df_mimic(sample_df, evaluation_alphas)