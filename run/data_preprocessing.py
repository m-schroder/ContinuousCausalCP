import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import pandas as pd
import numpy as np




def extract_mimic_data(config_data):
    vital_list = config_data["dynamic_cov"]
    static_list = config_data["static_cov"]

    data_path = "data/MIMIC/all_hourly_data.h5"
    h5 = pd.HDFStore(data_path, 'r')

    all_vitals = h5['/vitals_labs_mean'][vital_list]
    all_vitals = all_vitals.droplevel(['hadm_id', 'icustay_id'])
    static_features = h5['/patients'][static_list]
    static_features = static_features.droplevel(['hadm_id', 'icustay_id'])
    treatment = h5['/interventions'][['vent']]

    column_names_vitals = []
    for column in all_vitals.columns:
        if isinstance(column, str):
            column_names_vitals.append(column)
        else:
            column_names_vitals.append(column[0])
    all_vitals.columns = column_names_vitals


    # Filtering out users with time length < necissary time length
    user_sizes = all_vitals.groupby('subject_id').size()
    filtered_users_len = user_sizes.index[user_sizes > config_data["cov_window"] + config_data["treat_window"] + config_data["out_window"]]

    # Filtering out users with time age > 100
    if static_list is not None:
        if "age" in static_list:
            filtered_users_age = static_features.index[static_features.age < 100]
            filtered_users = filtered_users_len.intersection(filtered_users_age)
        else:
            filtered_users = filtered_users_len
    else:
        filtered_users = filtered_users_len


    #filtered_users = np.random.choice(filtered_users, size=n, replace=False)
    all_vitals = all_vitals.loc[filtered_users]

    # Split time-series into pre-treatment, treatmet, post-treatment part, and take mean -> static dataset
    vitals_pretreat = []
    vitals_out = []
    treatment_grouped = treatment.groupby('subject_id')
    treatment_pretreat = []
    treatment_treat = []
    for i, cov in enumerate(all_vitals.groupby('subject_id')):
        test = cov[1].to_numpy()
        T = test.shape[0]
        # sample random treatment time point
        t = np.random.randint(config_data["cov_window"], T-config_data["treat_window"] - config_data["out_window"])
        t_start = t - config_data["cov_window"]
        t_treat_end = t + config_data["treat_window"]


        # get ith treatment group as a numpy array
        treatment_i = treatment_grouped.get_group(cov[0]).to_numpy()

        treatment_pretreat.append(np.nanmean(treatment_i[t_start:t, :], axis=0, keepdims=True))
        treatment_treat.append(np.mean(treatment_i[t:t_treat_end, :], axis=0))

        t_out_end = t + config_data["treat_window"] + config_data["out_window"]
        vitals_pretreat.append(np.nanmean(test[t_start:t, :], axis=0, keepdims=True))
        vitals_out.append(np.nanmean(test[t_treat_end:t_out_end, :], axis=0, keepdims=True))



    vitals_pretreat = np.concatenate(vitals_pretreat, axis=0)
    vitals_pretreat = pd.DataFrame(vitals_pretreat, columns=column_names_vitals)
    # Set indices to subject_id
    vitals_pretreat.index = filtered_users
    vitals_out = np.concatenate(vitals_out, axis=0)
    vitals_out = pd.DataFrame(vitals_out, columns=column_names_vitals)
    vitals_out.index = filtered_users
    treatment_pretreat = np.concatenate(treatment_pretreat, axis=0)
    treatment_pretreat = pd.DataFrame(treatment_pretreat, columns=['vent'])
    treatment_pretreat.index = filtered_users
    treatment_treat = np.concatenate(treatment_treat, axis=0)
    treatment_treat = pd.DataFrame(treatment_treat, columns=['vent'])
    treatment_treat.index = filtered_users

    # One-hot encoding/ Standardization for static covariates
    static_features = static_features.loc[filtered_users]
    # Standardize age
    mean = np.mean(static_features["age"])
    std = np.std(static_features["age"])
    static_features["age"] = (static_features["age"] - mean) / std
    # set gender to 1 if "M", otherwise 0
    static_features["gender"] = np.where(static_features["gender"] == "M", 1, 0)

    # Get indices of rows with missing values
    idx = vitals_pretreat.index[vitals_pretreat.isnull().any(axis=1)]
    # Drop rows with missing values
    vitals_pretreat = vitals_pretreat.drop(idx)
    vitals_out = vitals_out.drop(idx)
    treatment_treat = treatment_treat.drop(idx)
    treatment_pretreat = treatment_pretreat.drop(idx)
    static_features = static_features.drop(idx)
    # Remove outliers
    for column in vitals_pretreat.columns:
        # Get indices of all rows that are below 0.1 percentile or above 99.9 percentile
        col = vitals_pretreat[column].to_numpy()
        quant_low = np.quantile(col, 0.001)
        quant_high = np.quantile(col, 0.999)
        # Get indices
        idx = vitals_pretreat.index[(col < quant_low) | (col > quant_high)]
        # Remove rows
        vitals_pretreat = vitals_pretreat.drop(idx)
        vitals_out = vitals_out.drop(idx)
        treatment_treat = treatment_treat.drop(idx)
        treatment_pretreat = treatment_pretreat.drop(idx)
        static_features = static_features.drop(idx)

    #Standardize covariates
    std_information = {}
    for column in vitals_pretreat.columns:
        mean = np.mean(vitals_pretreat[column])
        std = np.std(vitals_pretreat[column])
        vitals_pretreat[column] = (vitals_pretreat[column] - mean) / std
        vitals_out[column] = (vitals_out[column] - mean) / std
        std_information[column] = [mean, std]

    # Standardize

    # concat static features, vitals and treatments pre-treatment
    X = pd.concat([static_features, vitals_pretreat], axis=1, join="inner")
    A_full = treatment_treat.to_numpy().squeeze()
    # extract only blood pressure as Y
    Y_full = vitals_out.to_numpy()[:,2]
    h5.close()

    X_full = X.to_numpy()

 
    return X_full, A_full, Y_full, std_information



data_config = ({
    "cov_window": 10,
    "treat_window": 10,
    "out_window": 5,
    "train_propensities": False,
    "validation": False,
    "dynamic_cov": ["heart rate", "sodium", "mean blood pressure", "glucose", "hematocrit", "respiratory rate"],
    "static_cov": ["gender", "age"]
})


X_full, A_full, Y_full, info = extract_mimic_data(data_config)
gender = X_full[:,0]
age =  X_full[:,1]
heart_rate = X_full[:,2]
sodium = X_full[:,3]
pressure = X_full[:,4]
glucose = X_full[:,5]
hematocrit = X_full[:,6]
respiration = X_full[:,7]

Y_full = 0.1*gender + 0.1*age -0.1*heart_rate + 0.1*sodium + 0.2*pressure + 0.1*hematocrit + 0.1*respiration + A_full
m = np.mean(Y_full)
std = np.std(Y_full)
Y_full = (Y_full - m)/std

df = pd.DataFrame({"gender": gender, "age": age, "heart rate": heart_rate, "sodium": sodium, "blood pressure": pressure, "glucose": glucose, "hematocrit": hematocrit, "respiratory rate": respiration, "A": A_full, "Y": Y_full})
# shuffle rows of df
df = df.sample(frac = 1)
len_df = df.shape[0]

info = pd.DataFrame.from_dict(info)
info["Y"] = np.array([m, std])

df_train = df.iloc[0:int(0.6*len_df)]
df_val = df.iloc[int(0.6*len_df): int(0.7*len_df)]
df_cal = df.iloc[int(0.7*len_df): int(0.9*len_df)]
df_test = df.iloc[int(0.9*len_df):]

df_train.to_csv('../data/MIMIC/mimic_data_train.csv', index=False)
df_val.to_csv('../data/MIMIC/mimic_data_val.csv', index=False)
df_cal.to_csv('../data/MIMIC/mimic_data_cal.csv', index=False)
df_test.to_csv('../data/MIMIC/mimic_data_test.csv', index=False)
info.to_csv('../data/MIMIC/mimic_std_information.csv', index=False)
