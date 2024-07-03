import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

import pyro.distributions as dist
import pyro.distributions.transforms as T

pl.seed_everything(2024)


class CondNormalizingFlow(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        input_size = config["input_dim"]
        output_size = config["output_dim"]
        hidden_size = config["hidden_dim"]
        count_bins = config["count_bins"]

        dist_base = dist.Normal(torch.zeros(output_size), torch.ones(output_size))
        self.y_transform = T.conditional_spline(output_size, context_dim=input_size, hidden_dims=[hidden_size], count_bins=count_bins, bound = 1)
        self.dist_y_given_x = dist.ConditionalTransformedDistribution(dist_base, [self.y_transform])

        self.optimizer = torch.optim.SGD(self.y_transform.parameters(), lr=config["lr"], momentum=0.9)
        self.save_hyperparameters()

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, batch):
        x, y = batch
        # Forward pass
        ln_p_y_given_x = self.dist_y_given_x.condition(x).log_prob(y.unsqueeze(dim=1))
        return {"obj": - ln_p_y_given_x.mean()}

    def training_step(self, train_batch, batch_idx):
        self.train()
        #loss
        obj_dict = self.forward(train_batch)
        return obj_dict["obj"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.dist_y_given_x.clear_cache()

    def validation_step(self, val_batch, batch_idx):
        self.eval()
        # Loss
        obj_val = self.forward(val_batch)
        return obj_val["obj"]

    #Evaluates density of y given x on a grid of y values
    #x is of shape (batch_size, d_in), y is of shape (n_grid, 1)
    def predict_density(self, x, y, scaling_params=None):
        self.eval()
        if scaling_params is not None:
            y = (y - scaling_params["mean"]) / scaling_params["sd"]
        pred = self.dist_y_given_x.condition(x).log_prob(y).exp()
        if scaling_params is not None:
            pred = pred / scaling_params["sd"]
        return pred

    def sample(self, x, n_samples, scaling_params=None):
        self.eval()
        # x is of shape (batch_size, d_in)
        samples = torch.squeeze(self.dist_y_given_x.condition(x).sample(torch.Size([n_samples, x.shape[0]])))
        if samples.dim() == 1:
            samples = samples.unsqueeze(1)
        samples = torch.transpose(samples, 0, 1)
        if scaling_params is not None:
            samples = samples * scaling_params["sd"] + scaling_params["mean"]
        return samples
    


class SimpleRegressionModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.1):
        super(SimpleRegressionModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.criterion = nn.MSELoss()
        
        self.save_hyperparameters()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs.squeeze(), targets)
        self.log('train_loss', loss, logger = True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        val_loss = self.criterion(outputs.squeeze(), targets)
        self.log('val_loss', val_loss, logger = True)
        #print(val_loss)
        return {'loss': val_loss}
    
    #MC-dropout intervals

    def predict_step(self, inputs):
        self.dropout.train()            # turn on MC dropout
        output_list = []
        for i in range(500):
            output = self(inputs).detach().numpy()
            output_list.append(output)
        quantile_list = np.quantile(a=output_list, q = np.array([0.005, 0.025, 0.05, 0.1, 0.25, 0.75, 0.9, 0.95, 0.975, 0.995]))    
        return quantile_list



class RealWorldRegressor(pl.LightningModule):
    def __init__(self, 
                 input_size,
                 hidden_dim,
                 dropout_rate = 0.5):
        super().__init__()
        self.n_features = input_size
        self.hidden_dim = hidden_dim
        self.criterion = nn.MSELoss()
        self.save_hyperparameters()

        self.net = nn.Sequential(
            nn.BatchNorm1d(self.n_features),
            nn.Linear(self.n_features, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim, 1)
            ) 

    def forward(self, x):
        out = self.net(x)
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat.squeeze(), y)
        self.log("train_loss", loss, logger = True, prog_bar=False)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat.squeeze(), y)
        self.log("val_loss", loss, logger = True, prog_bar=True)
        return {"loss": loss}

    def predict_step(self, inputs):
        self.dropout.train()            # turn on MC dropout
        output_list = []
        for i in range(500):
            output = self(inputs).detach().numpy()
            output_list.append(output)
        quantile_list = np.quantile(a=output_list, q = np.array([0.005, 0.025, 0.05, 0.1, 0.25, 0.75, 0.9, 0.95, 0.975, 0.995]))    
        return quantile_list