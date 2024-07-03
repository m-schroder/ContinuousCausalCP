import os, random
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np




# Covariate, propensity and outcome functions

def covariates(n_samples):
    return np.random.randint(1, 5, size=n_samples) 

def normal_prop(n_samples, cov):
    return np.random.normal(loc=5*cov, scale=10, size=n_samples)

def step_prop(n_samples, cov):
    p = np.random.binomial(1, 0.3, size=n_samples)
    return p*np.random.uniform(low=0, high=5*cov, size=n_samples) + (1-p)*np.random.uniform(low=5*cov, high=40, size=n_samples)

def concave_reponse(n_samples, treatment, cov):
    noise = np.random.normal(loc=0, scale=0.1, size=n_samples)
    return np.sin((0.1*treatment - 0.5*cov) *np.pi/6) + noise

def sin_response(n_samples, treatment, cov):
    noise = np.random.normal(loc=0, scale=0.1, size=n_samples)
    return np.sin((0.1*treatment - 0.1*cov) *np.pi/2) + noise

def exp_response(n_samples, treatment, cov):
    noise = np.random.normal(loc=0, scale=2, size=n_samples)
    return np.exp(0.1*treatment - 0.1*cov) + noise






# Dataset configurations per synthetic experiment 


def config_data1():     

    dataset_config = dict(
        #sample_sizes
        n_train = 2000,
        n_cal = 1000,
        n_test = 400,
        n_int = 1000,
        # distribution specification
        X = covariates,
        A = step_prop,
        Y = concave_reponse,
        # intervention specification
        delta1 = 1,
        delta5 = 5,
        delta10 = 10
    )
    model_config = dict(
        input_dim = 2,
        output_dim = 1, 
        hidden_dim = 16,
        epochs = 300 
    )
    density_config = dict(              
        input_dim = 8,
        output_dim = 1, 
        hidden_dim = 64,
        count_bins = 12,
        lr = 1e-3,
        neptune = True
    )

    return dataset_config, model_config, density_config



def config_data2():         

    dataset_config = dict(
        #sample_sizes
        n_train = 2000,
        n_cal = 1000,
        n_test = 400,
        n_int = 1000,
        # distribution specification
        X = covariates,
        A = normal_prop,
        Y = sin_response,
        # intervention specification
        delta1 = 1,
        delta5 = 5,
        delta10 = 10
    )
    model_config = dict(
        input_dim = 2,
        output_dim = 1, 
        hidden_dim = 16,
        epochs = 300 
    )
    density_config = dict(             
        input_dim = 8,
        output_dim = 1, 
        hidden_dim = 64,
        count_bins = 12,
        lr = 1e-3,
        neptune = True
    )

    return dataset_config, model_config, density_config