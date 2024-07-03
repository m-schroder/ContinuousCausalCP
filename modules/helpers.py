import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so





def plot_intervals(df, path = None):
    
    colorplt = sns.color_palette()
    sns.set_style("dark")
    sns.set(font_scale = 1.1)

    g = sns.FacetGrid(df, col = "X", row = "α", margin_titles=True, gridspec_kws={"wspace":0.1, "hspace": -0.15})
    g.map_dataframe(sns.scatterplot, x="A", y="Y", label = "True outcome", color=".2", marker = ".", edgecolor = None)
    for (alpha,x), ax in g.axes_dict.items():
        ax.fill_between(x = df.loc[(df["α"]== alpha) & (df["X"]== x), "A"], y1=df.loc[(df["α"]== alpha) & (df["X"]== x), "C_low_mc"], y2=df.loc[(df["α"]== alpha) & (df["X"]== x), "C_up_mc"], color=colorplt[1], alpha=0.6, label='MC interval', zorder=2)
        ax.fill_between(x = df.loc[(df["α"]== alpha) & (df["X"]== x), "A"], y1=df.loc[(df["α"]== alpha) & (df["X"]== x), "C_low"], y2=df.loc[(df["α"]== alpha) & (df["X"]== x), "C_up"], color=colorplt[0], alpha=0.5, label='Ours', zorder=1)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.25, box.width, box.height * 0.75])

    ax.legend(loc='upper center', bbox_to_anchor=(-1.2, -0.5), ncol = 3)

    g.set_xlabels("Treatment")
    g.set_ylabels("Outcome")

    if path!= None:
        g.savefig(path, bbox_inches='tight')
    plt.show() 


def evaluation_boxplots(df, dataset, metric, path = None):

    df_small = df.loc[df["dataset"]==dataset]
    df_delta_1 = df_small.loc[df_small["delta"]==1]
    df_delta_5 = df_small.loc[df_small["delta"]==5]
    df_delta_10 = df_small.loc[df_small["delta"]==10]

    df_plot = pd.DataFrame(columns=("seed", "delta", "alpha", "coverage", "coverage_mc", "length", "length_mc"))


    alphas = [0.05, 0.1, 0.2]
    seeds = [2024,0,1,2,3,4,5,6,7,8]

    for alpha in alphas:
        for seed in seeds:
            df_seed_1 = df_delta_1.loc[df_delta_1["seed"]==seed]
            coverage_1 = np.sum(df_seed_1.loc[df_seed_1["alpha"]==alpha, "coverage"])/df_seed_1.loc[df_seed_1["alpha"]==alpha].shape[0]
            avg_length = np.sum(df_seed_1.loc[df_seed_1["alpha"]==alpha, "length"])/df_seed_1.loc[df_seed_1["alpha"]==alpha].shape[0]
            coverage_1_mc = np.sum(df_seed_1.loc[df_seed_1["alpha"]==alpha, "coverage_mc"])/df_seed_1.loc[df_seed_1["alpha"]==alpha].shape[0]
            avg_length_mc = np.sum(df_seed_1.loc[df_seed_1["alpha"]==alpha, "length_mc"])/df_seed_1.loc[df_seed_1["alpha"]==alpha].shape[0]
            df_plot = pd.concat([df_plot, pd.DataFrame({"seed": [seed], "delta": [1], "alpha": [alpha], "coverage": [coverage_1], "coverage_mc": [coverage_1_mc], "length": [avg_length], "length_mc": [avg_length_mc]})])

            df_seed_2 = df_delta_5.loc[df_delta_5["seed"]==seed]
            coverage_2 = np.sum(df_seed_2.loc[df_seed_2["alpha"]==alpha, "coverage"])/df_seed_2.loc[df_seed_2["alpha"]==alpha].shape[0]
            avg_length_2 = np.sum(df_seed_2.loc[df_seed_2["alpha"]==alpha, "length"])/df_seed_2.loc[df_seed_2["alpha"]==alpha].shape[0]
            coverage_2_mc = np.sum(df_seed_2.loc[df_seed_2["alpha"]==alpha, "coverage_mc"])/df_seed_2.loc[df_seed_2["alpha"]==alpha].shape[0]
            avg_length_2_mc = np.sum(df_seed_2.loc[df_seed_2["alpha"]==alpha, "length_mc"])/df_seed_2.loc[df_seed_2["alpha"]==alpha].shape[0]
            df_plot = pd.concat([df_plot, pd.DataFrame({"seed": [seed], "delta": [5], "alpha": [alpha], "coverage": [coverage_2], "coverage_mc": [coverage_2_mc], "length": [avg_length_2], "length_mc": [avg_length_2_mc]})])

            df_seed_3 = df_delta_10.loc[df_delta_10["seed"]==seed]
            coverage_3 = np.sum(df_seed_3.loc[df_seed_3["alpha"]==alpha, "coverage"])/df_seed_3.loc[df_seed_3["alpha"]==alpha].shape[0]
            avg_length_3 = np.sum(df_seed_3.loc[df_seed_3["alpha"]==alpha, "length"])/df_seed_3.loc[df_seed_3["alpha"]==alpha].shape[0]
            coverage_3_mc = np.sum(df_seed_3.loc[df_seed_3["alpha"]==alpha, "coverage_mc"])/df_seed_3.loc[df_seed_3["alpha"]==alpha].shape[0]
            avg_length_3_mc = np.sum(df_seed_3.loc[df_seed_3["alpha"]==alpha, "length_mc"])/df_seed_3.loc[df_seed_3["alpha"]==alpha].shape[0]
            df_plot = pd.concat([df_plot, pd.DataFrame({"seed": [seed], "delta": [10], "alpha": [alpha], "coverage": [coverage_3], "coverage_mc": [coverage_3_mc], "length": [avg_length_3], "length_mc": [avg_length_3_mc]})])

    df_plot = df_plot.melt(id_vars=["seed", "delta", "alpha"])
    df_plot = df_plot.rename({"alpha":'α'}, axis=1)

    sns.set_style("dark")
    sns.set(font_scale = 1.2)
    colorplt = sns.color_palette(palette="pastel")

    if metric == "faithfulness":
        df_plot = df_plot.loc[df_plot["variable"]!="length"]
        df_plot = df_plot.loc[df_plot["variable"]!="length_mc"]
        df_plot = df_plot.replace("coverage_mc","MC dropout")
        df_plot = df_plot.replace("coverage","Ours")
        df_plot = df_plot.sort_values(by = "variable")

        g = sns.FacetGrid(df_plot, col = "α")
        g.map_dataframe(sns.boxplot, x="value", y="variable", hue = "delta", palette = colorplt) 
        for alpha, ax in g.axes_dict.items():
            ax.axvline(1-alpha, color = "indianred", ls = "--", lw = 2)
        g.set_xlabels("Coverage", fontsize = 15)
        g.set_ylabels("")
        g.add_legend(title = "$\Delta_a$")

    elif metric == "sharpness":
        df_plot = df_plot.loc[df_plot["variable"]!="coverage"]
        df_plot = df_plot.loc[df_plot["variable"]!="coverage_mc"]
        df_plot = df_plot.replace("length_mc", "MC dropout")
        df_plot = df_plot.replace("length", "Ours")
        df_plot = df_plot.sort_values(by = "variable")

        g = sns.FacetGrid(df_plot, col = "α", margin_titles=True)
        g.map_dataframe(sns.boxplot, x="value", y="variable", hue = "delta", palette = colorplt)
        g.set_xlabels("Interval length", fontsize = 15)
        g.set_ylabels("")
        g.add_legend(title = "$\Delta_a$")

    else:
        raise ValueError

    if path!= None:
        plt.savefig(path)
    plt.show() 


def mimic_plots(patient_1, patient_2, patient_3, patient_4, path = None):

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 4))
    fig.tight_layout()
    sns.set(style = "darkgrid")
    cl = sns.color_palette("Paired")
    x = np.arange(0, 1.1, 0.1)

    ax1.fill_between(x, patient_1["C_low"], patient_1["C_up"], alpha = 0.6, label = "Young", zorder=2, color = cl[1])
    ax1.fill_between(x,  patient_2["C_low"], patient_2["C_up"], alpha = 0.6, label = "Old",zorder=1,  color = cl[0])
    ax1.plot(x, patient_1["Y_pred"], '--', color=".3")
    ax1.plot(x, patient_2["Y_pred"], '--', color=".5")
    ax1.set_ylim(50,150)
    ax1.set_xlim(0,1)
    ax1.legend(loc='upper left', title = "Male")
    ax1.set_ylabel("Blood pressure")

    ax2.fill_between(x, patient_3["C_low"], patient_3["C_up"], alpha = 0.5, label = "Young", zorder=2, color = cl[9])
    ax2.fill_between(x,  patient_4["C_low"], patient_4["C_up"], alpha = 0.5, label = "Old",zorder=1, color = cl[8])
    ax2.plot(x, patient_3["Y_pred"], '--', color=".3")
    ax2.plot(x, patient_4["Y_pred"], '--', color=".5")
    ax2.set_ylim(50,150)
    ax2.set_xlim(0,1)
    ax2.legend(loc='upper left', title = "Female")
    ax2.set_ylabel("Blood pressure")
    ax2.set_xlabel("Duration of ventilation (scaled)")
    
    if path!= None:
        plt.savefig(path)
    plt.show() 
