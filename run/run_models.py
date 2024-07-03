import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np

from modules.prediction_models import CondNormalizingFlow, RealWorldRegressor, SimpleRegressionModel

from modules import dataset_configs

pl.seed_everything(2024)


save_location = "../models"

# A) MIMIC normalizing flow
model_name = "Denisty_estimator_mimic"

data_train_mimic = pd.read_csv('../data/MIMIC/mimic_data_train.csv').dropna()
data_val_mimic = pd.read_csv('../data/MIMIC/mimic_data_val.csv').dropna()

X_train = torch.tensor(data_train_mimic[['gender', 'age', 'heart rate', 'sodium', 'blood pressure', 'glucose', 'hematocrit', 'respiratory rate']].values, dtype=torch.float32)
y_train = torch.tensor(data_train_mimic['A'].values, dtype=torch.float32)
dataset_train = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)

X_val = torch.tensor(data_val_mimic[['gender', 'age', 'heart rate', 'sodium', 'blood pressure', 'glucose','hematocrit', 'respiratory rate']].values, dtype=torch.float32)
y_val = torch.tensor(data_val_mimic['A'].values, dtype=torch.float32)
dataset_val = TensorDataset(X_val, y_val)
val_loader = DataLoader(dataset_val, batch_size=32, shuffle=True)

config = {
    "input_dim":8,
    "output_dim":1,
    "hidden_dim":64,
    "count_bins":10,
    "lr": 1e-4
}

model = CondNormalizingFlow(config)
logger = TensorBoardLogger(save_dir="logs/")
checkpoint_callback = ModelCheckpoint( monitor="train_loss", dirpath= save_location + "/" + model_name, filename= model_name + "_checkpoints", save_top_k=1, mode="min",)
trainer = pl.Trainer(max_epochs=50)
trainer.fit(model, train_loader, val_loader)
trainer.save_checkpoint(save_location + "/" + model_name + "_checkpoints.ckpt")




# B) MIMIC regressor

seed_list = [2024, 0 ,1,2,3,4,5,6,7,8]
loss_list = []

for seed in seed_list:

    pl.seed_everything(seed)
    model_name = "Regression_mimic_" + str(seed)

    X_train = torch.tensor(data_train_mimic[['gender', 'age', 'heart rate', 'sodium', 'blood pressure', 'glucose', 'hematocrit', 'respiratory rate', 'A']].values, dtype=torch.float32)
    y_train = torch.tensor(data_train_mimic['Y'].values, dtype=torch.float32)
    dataset_train = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)

    X_val = torch.tensor(data_val_mimic[['gender', 'age', 'heart rate', 'sodium', 'blood pressure', 'glucose', 'hematocrit', 'respiratory rate', 'A']].values, dtype=torch.float32)
    y_val = torch.tensor(data_val_mimic['Y'].values, dtype=torch.float32)
    dataset_val = TensorDataset(X_val, y_val)
    val_loader = DataLoader(dataset_val, batch_size=32, shuffle=True)

    model = RealWorldRegressor(input_size=9, hidden_dim=128)
    logger = TensorBoardLogger(save_dir="logs/")
    checkpoint_callback = ModelCheckpoint( monitor="train_loss", dirpath= save_location + "/" + model_name, filename= model_name + "_checkpoints", save_top_k=1, mode="min",)
    trainer = pl.Trainer(max_epochs=15, logger = logger, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint(save_location + "/" + model_name + "_checkpoints.ckpt")

    test_df = pd.read_csv('data/MIMIC/mimic_data_test.csv').dropna()  
    X_test = torch.tensor(test_df[['gender', 'age', 'heart rate', 'sodium', 'blood pressure', 'glucose', 'hematocrit', 'respiratory rate', 'A']].values, dtype=torch.float32)
    y_test = torch.tensor(test_df['Y'].values, dtype=torch.float32)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=test_df.shape[0])

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            predicted_outcomes = outputs.squeeze().numpy()
            loss = nn.MSELoss()(outputs.squeeze(), targets)
            test_loss += loss.item() * len(inputs)

    avg_test_loss = test_loss / len(test_dataset)
    loss_list.append(avg_test_loss)

print("Mean test Loss MIMIC:", np.mean(loss_list))
print("Std test Loss MIMIC:", np.std(loss_list))





# C) Regressor synthetic datasets

seed_list = [2024, 0 ,1,2,3,4,5,6,7,8]
dataset_list = [2,3]
configs = [dataset_configs.config_data1(), dataset_configs.config_data2()]

for dataset in dataset_list:
    
   dataset_config, model_config, _ = configs[dataset-2]

   for seed in seed_list:

        model_name = "Regression_simulated_" + str(dataset) + "_seed" + str(seed)
        model = SimpleRegressionModel(input_size=model_config["input_dim"], hidden_size=model_config["hidden_dim"], output_size=model_config["output_dim"])

        df = pd.read_csv('../data/simulated_data' + str(dataset) + '.csv').sample(frac=1).reset_index(drop=True) 
        df_train = df.loc[0:0.8*df.shape[0],:]
        df_val = df.loc[0.8*df.shape[0]:,:]

        X_train = torch.tensor(df_train[['X', 'A']].values, dtype=torch.float32)
        y_train = torch.tensor(df_train['Y'].values, dtype=torch.float32)
        dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        X_val = torch.tensor(df_train[['X', 'A']].values, dtype=torch.float32)
        y_val = torch.tensor(df_train['Y'].values, dtype=torch.float32)
        dataset_val = TensorDataset(X_val, y_val)
        val_loader = DataLoader(dataset_val, batch_size=32, shuffle=True)

        logger = TensorBoardLogger(save_dir="logs/")
        trainer = pl.Trainer(max_epochs=300, logger = logger)
        trainer.fit(model, train_loader, val_loader)
        trainer.save_checkpoint(save_location + "/" + model_name + "_checkpoints.ckpt")


df_test = pd.read_csv('data/simulated_data' + str(dataset) + '_test.csv')
X_test = torch.tensor(df_test[['X', 'A']].values, dtype=torch.float32)
y_test = torch.tensor(df_test['Y'].values, dtype=torch.float32)

for dataset in dataset_list:
    mse_list = []
    for seed in seed_list:
        if seed == 2024:
            model = SimpleRegressionModel.load_from_checkpoint('../models/Regression_simulated_' + str(dataset) +  '_checkpoints.ckpt').eval()
        else:
            model = SimpleRegressionModel.load_from_checkpoint('../models/Regression_simulated_' + str(dataset) + '_seed'+ str(seed) +  '_checkpoints.ckpt').eval()
        with torch.no_grad():
            preds = model(X_test)
            loss = nn.MSELoss()
            mse = loss(preds, y_test.unsqueeze(dim=1)).detach().numpy()
            
        mse_list.append(mse)
    m = np.mean(mse_list)
    s = np.std(mse_list)
    print("Mean MSE Dataset " + str(dataset) + ": " + str(m))
    print("Std MSE Dataset " + str(dataset) + ": " + str(s))