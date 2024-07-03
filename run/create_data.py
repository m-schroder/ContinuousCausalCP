import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modules import dataset_configs

np.random.seed(2024)



"""
    State which dataset from configs should be created
""" 

dataset_number = 3

dataset_list = [dataset_configs.config_data1(), dataset_configs.config_data2()]
config = dataset_list[dataset_number-1][0]


n_train = config["n_train"]
n_test = config["n_cal"]
n_cal = config["n_test"]
n_intervention = config["n_int"]



"""
    Create dataframes
""" 

# train
X = config["X"](n_train)
A = config["A"](n_train, X)
df_train = pd.DataFrame({
    'X': X,
    'A': A,
    'Y': config["Y"](n_train, A, X)
})
df_train.to_csv('../data/simulated_data' + str(dataset_number) + '.csv', index=False) 


# calibration
X = config["X"](n_cal)
A = config["A"](n_cal, X)
df_cal = pd.DataFrame({
    'X': X,
    'A': A,
    'Y': config["Y"](n_cal, A, X)
})
df_cal.to_csv('../data/simulated_data' + str(dataset_number) + '_calibration.csv', index=False) 


# test
X = config["X"](n_test)
A = config["A"](n_test, X)
df_test = pd.DataFrame({
    'X': X,
    'A': A,
    'Y': config["Y"](n_test, A, X)
})
df_test.to_csv('../data/simulated_data' + str(dataset_number) + '_test.csv', index=False) 




"""
Create interventional data
"""

#soft intervention
for delta in [1,5,10]:

    X = config["X"](n_intervention)
    A = config["A"](n_intervention, X+(delta/5))        # set intervention as +1, +5, +10

    df = pd.DataFrame({
        'X': X,
        'A': A,
        'Y': config["Y"](n_intervention, A, X)
    })

    df_sorted = df.sort_values(by="A", axis = 0)

    df_X1 = df_sorted.loc[df_sorted['X'] ==1]
    df_X2 = df_sorted.loc[df_sorted['X'] ==2]
    df_X3 = df_sorted.loc[df_sorted['X'] ==3]
    df_X4 = df_sorted.loc[df_sorted['X'] ==4]

    df_X1.to_csv('../data/simulated_data' + str(dataset_number) + '_Delta' + str(delta) + '_X1.csv', index=False) 
    df_X2.to_csv('../data/simulated_data' + str(dataset_number) + '_Delta' + str(delta) + '_X2.csv', index=False) 
    df_X3.to_csv('../data/simulated_data' + str(dataset_number) + '_Delta' + str(delta) + '_X3.csv', index=False) 
    df_X4.to_csv('../data/simulated_data' + str(dataset_number) + '_Delta' + str(delta) + '_X4.csv', index=False) 



# hard intervention
for intervention in [5,7,10]:
        
    X = config["X"](n_intervention)
    A = intervention*X

    df = pd.DataFrame({
        'X': X,
        'A': A,
        #'A_star': A_star,
        'Y': config["Y"](n_intervention, A, X)
    })

    df_sorted = df.sort_values(by="X", axis = 0)
    df_sorted.to_csv('../data/simulated_data' + str(dataset_number) + '_Intervention' + str(intervention) + '.csv', index=False) 

    df_X1 = df_sorted.loc[df_sorted['X'] ==1]
    df_X2 = df_sorted.loc[df_sorted['X'] ==2]
    df_X3 = df_sorted.loc[df_sorted['X'] ==3]
    df_X4 = df_sorted.loc[df_sorted['X'] ==4]

    df_X1.to_csv('../data/simulated_data' + str(dataset_number) + '_Intervention' + str(intervention) + '_X1.csv', index=False) 
    df_X2.to_csv('../data/simulated_data' + str(dataset_number) + '_Intervention' + str(intervention) + '_X2.csv', index=False) 
    df_X3.to_csv('../data/simulated_data' + str(dataset_number) + '_Intervention' + str(intervention) + '_X3.csv', index=False) 
    df_X4.to_csv('../data/simulated_data' + str(dataset_number) + '_Intervention' + str(intervention) + '_X4.csv', index=False) 


