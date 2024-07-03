import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import pandas as pd
import numpy as np

from modules.helpers import plot_intervals, evaluation_boxplots, mimic_plots


# Plot prediction intervals (unknown propensity)

df= pd.read_csv('../results/prediction_intervals_simulated_known.csv') 
df = df.rename({"alpha":'α'}, axis=1)

for dataset in np.unique(df["dataset"]): 
    for delta in np.unique(df["delta"]):
        df_plot = df.loc[df["dataset"] == dataset]
        df_plot = df_plot.loc[df["delta"] == delta]
        df_plot = df_plot.loc[df_plot["A"] >= 0]
        df_plot = df_plot.loc[df_plot["A"] <= 40]
        df_plot= df_plot.loc[df_plot["α"] != 0.01]  

        plot_intervals(df_plot, path="../results/plots/interval_facets_dataset" + str(dataset) + "_delta" + str(delta) + ".pdf")


# Plot boxplots for coverage and length

df= pd.read_csv('../results/intervals_simulated_known_seeds.csv') 

for dataset in np.unique(df["dataset"]):
    evaluation_boxplots(df, dataset, "faithfulness", "../results/plots/dataset" + str(dataset) + "_coverage.pdf")
    evaluation_boxplots(df, dataset, "sharpness", "../results/plots/dataset" + str(dataset) + "_length.pdf")


# Plot mimic intervals

df_plot = pd.read_csv('../results/prediction_intervals_mimic.csv')
info = pd.read_csv('../data/MIMIC/mimic_std_information.csv')

# rescaling
df_plot["Y"] = df_plot["Y"]*info["mean blood pressure"][1] + info["mean blood pressure"][0]
df_plot["C_low"] = df_plot["C_low"]*info["mean blood pressure"][1] + info["mean blood pressure"][0]
df_plot["C_up"] = df_plot["C_up"]*info["mean blood pressure"][1] + info["mean blood pressure"][0]
df_plot["Y_pred"] = df_plot["Y_pred"]*info["mean blood pressure"][1] + info["mean blood pressure"][0]

# specify patients to plot
patient_1 = df_plot.loc[(df_plot["gender"] == 1) & (df_plot["age"] < 0)][0]
patient_2 = df_plot.loc[(df_plot["gender"] == 1) & (df_plot["age"] > 0)][0]
patient_3 = df_plot.loc[(df_plot["gender"] == 0) & (df_plot["age"] < 0)][0]
patient_4 = df_plot.loc[(df_plot["gender"] == 0) & (df_plot["age"] > 0)][0]

mimic_plots(patient_1, patient_2, patient_3, patient_4, "../results/plots/intervals_mimic.pdf")