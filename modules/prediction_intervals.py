import os
import sys
module_path = os.path.abspath(os.path.join('..'))

import timeit

if module_path not in sys.path:
    sys.path.append(module_path)



import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import cvxpy as cp
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from modules.prediction_models import SimpleRegressionModel, CondNormalizingFlow, RealWorldRegressor


np.random.seed(2024) 
pl.seed_everything(2024)


#---------------------------------------------------------------- Define class for calculating prediction intervals----------------------------------

class prediction_intervals:

    def __init__(self, 
                 model,
                 calibration_data, 
                 new_sample,
                 intervention_mode,
                 alpha,
                 eps,
                 M = None,
                 intervention = None,
                 delta_A = None,
                 prop_estimator = None,
                 seed = None, 
                 task = None):
        self.model = model.eval()
        self.calibration_data = calibration_data
        self.new_sample = new_sample

        self.intervention_mode = intervention_mode
        self.intervention = intervention
        self.delta_A = delta_A
        self.prop_estimator = prop_estimator
        self.alpha = alpha
        self.M = M
        self.eps = eps
        self.task = task

        if seed == None:
            np.random.seed(2024) 
            pl.seed_everything(2024)
        else:
            np.random.seed(seed) 
            pl.seed_everything(seed)

        with torch.no_grad():
            model.eval()
            self.conf_scores = prediction_intervals.conformity_score(self, y = self.calibration_data["Y"].values.reshape(-1,1), y_hat = self.model(torch.Tensor(self.calibration_data.loc[:, self.calibration_data.columns != "Y"].values)).detach().numpy())

        self.all_data = pd.concat((self.calibration_data.loc[:, self.calibration_data.columns != "Y"], self.new_sample.drop("Y", axis=1)), ignore_index=True)

        self.all_data["A"] = self.all_data["A"]
        self.n = self.calibration_data.shape[0]


    def conformity_score(self, y, y_hat):
        return np.abs(y-y_hat)#.squeeze()

    def propensity_func(a,x, dataset):
        if dataset == 1:   
            return 1/(10*np.sqrt(2* np.pi))*np.exp(-(a - 5*x)**2/200)
        else:
            return np.where(a < 5*x, 0.3/(5*x), 0.7/(40-5*x))


    def optimize(self, S):

        if self.intervention_mode == "unknown":   

            a_x = self.all_data.to_numpy()
            S_impute = np.concatenate((self.conf_scores, S.reshape(-1,1))).squeeze()
            
            if self.task == "mimic":
                propensity_estimates =  self.prop_estimator.predict_density(torch.Tensor(self.all_data.loc[:, self.all_data.columns !="A"].values), torch.Tensor(self.all_data["A"].values).unsqueeze(dim=1)).detach().numpy()
                propensity_estimates[propensity_estimates < 0.05] = 0.05 
                a_star = np.repeat(self.new_sample["A"], self.n+1)
            else:
                propensity_estimates = prediction_intervals.propensity_func(a_x[:,1],a_x[:,0], self.task)#.reshape(self.n +1, 1)
                a_star = (self.all_data["X"]*self.intervention)#.values.reshape(self.n +1, 1)
            diff_treatment = np.array(np.abs(a_x[:,1] - a_star))#.reshape(self.n +1, 1)

            opt_model = pyo.ConcreteModel()
            opt_model.sigma = pyo.Var(domain=pyo.PositiveReals, bounds=(0, 1))  
            opt_model.c_a = pyo.Var(domain=pyo.PositiveReals, bounds=(1/self.M, self.M))  
            opt_model.set = pyo.RangeSet(self.n + 1)
            opt_model.u = pyo.Var(opt_model.set, domain=pyo.NonNegativeReals)  
            opt_model.v = pyo.Var(opt_model.set, domain=pyo.NonNegativeReals)  
            opt_model.OBJ = pyo.Objective(expr = sum(((1-self.alpha)*opt_model.u[i] + self.alpha*opt_model.v[i]) for i in opt_model.u))

            opt_model.Constraint =pyo.ConstraintList()
            for i in opt_model.u:
                opt_model.Constraint.add(expr = opt_model.u[i] - opt_model.v[i] - S_impute[i-1] + opt_model.c_a/np.sqrt(2*np.pi) * 1/propensity_estimates[i-1] * 1/opt_model.sigma * pyo.exp(-np.square(diff_treatment[i-1] * opt_model.sigma)/2) == 0)

            opt_model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT) 

            solver = SolverFactory('ipopt')
            solver.solve(opt_model)
            res = solver.solve(opt_model)
            print(res.Solver.status)

            out = opt_model.v.extract_values()[self.n+1]

            return out
        

        else:
            a_x = self.all_data.to_numpy()
            propensities = self.propensity_func(a_x[:,1],a_x[:,0], self.task)
            soft_intervention = self.propensity_func(a_x[:,1] - np.repeat(self.delta_A, self.n + 1), a_x[:,0], self.task)
            S_impute = cp.Constant(np.concatenate((self.conf_scores, S)))
            prop_ratio = cp.Constant(soft_intervention/propensities)

            # define variables
            theta = cp.Variable(shape=(1,1), pos=True)
            u = cp.Variable(self.n+1, nonneg = True)
            v = cp.Variable(self.n+1, nonneg = True)

            # Create constraints
            constraints = [cp.vec(S_impute) - cp.multiply(theta, cp.vec(prop_ratio))-u+v == cp.Constant(np.zeros(self.n +1))]

            # state optimization objective
            obj = cp.Minimize(cp.sum(cp.multiply(cp.Constant(1-self.alpha),u) + cp.multiply(cp.Constant(self.alpha),v)))

            # solve problem
            prob = cp.Problem(obj, constraints)
            try:
                prob.solve(warm_start = True)
            except Exception as e:
                print(e)

            eta = constraints[0].dual_value[-1]

            return eta
        

    def binary_search(self):
        
        S_up = np.max(self.conf_scores)
        S_low = np.array([0])

        if self.intervention_mode == "known":

            eta_up = prediction_intervals.optimize(self, S_up)
            eta_low = prediction_intervals.optimize(self, S_low)
            while eta_up < 1 - self.alpha - np.exp(-3):
                S_up = 2*S_up
                eta_up = prediction_intervals.optimize(self, S_up)
            while eta_low == 1-self.alpha:
                S_low = 1/2*S_low
                eta_low = prediction_intervals.optimize(self, S_low)  

            S_final = (S_up + S_low)/2  
            while S_up - S_low > self.eps:   
                eta_S = prediction_intervals.optimize(self, S_final)
                if eta_S < 1-self.alpha - np.exp(-3):
                    S_low = (S_up + S_low)/2
                else:
                    S_up = (S_up + S_low)/2
                S_final = (S_up + S_low)/2

        elif self.intervention_mode == "unknown":

            v_up = prediction_intervals.optimize(self, S_up)
            v_low = prediction_intervals.optimize(self, S_low)

            while v_up > 0: 
                S_up = 2*S_up
                v_up = prediction_intervals.optimize(self, S_up)
            while v_low ==0: 
                S_low = 1/2*S_low
                v_low = prediction_intervals.optimize(self, S_low)  

            S_final = (S_up + S_low)/2  
            while S_up - S_low > self.eps:   
                v_S = prediction_intervals.optimize(self, S_final)
                if v_S > 0: 
                    S_low = (S_up + S_low)/2
                else:
                    S_up = (S_up + S_low)/2
                S_final = (S_up + S_low)/2

        else:
            raise KeyError
        
        return S_final 



    def create_interval(self):

        self.model.eval()
        with torch.no_grad():
            if self.task == "mimic":
                tensor = torch.Tensor(self.new_sample.loc[:, self.new_sample.columns != "Y"].values)
            else:
                tensor = torch.Tensor(self.new_sample.loc[:, self.new_sample.columns != "Y"].values[0])
            y_hat = self.model(tensor).detach().numpy()[0]    
        S_star = prediction_intervals.binary_search(self)

        C_low = y_hat - S_star
        C_up = y_hat + S_star

        return C_low, C_up #C_low[0], C_up[0]








#------------------------------------------------------------------ Calculate intervals -------------------------------------------------------------


def generate_df_mimic(test_data, evaluation_alphas):
    '''
        First need to have generated an empty dataframe via 
        df = pd.DataFrame(columns=["a_star", "alpha", "eps", "M", "gender", "age", "heart rate", "sodium", "blood pressure", "glucose", "hematocrit", "respiratory rate", "A", "Y", "Y_pred", "C_low", "C_up", "C_low_mc", "C_up_mc"])
        and saved it at
        df.to_csv('results/prediction_intervals_mimic.csv', index=False) 
    '''
    alpha_list = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 0.5, 0.2, 0.1, 0.05, 0.01])        # MC- quantiles outputed by prediction step in regression model
    eps = 0.0001
    M = 3

    df_intervals = pd.read_csv('results/prediction_intervals_mimic_semi.csv') 

    calibration_data = pd.read_csv('data/MIMIC/mimic_data_cal.csv').dropna() 
    #calibration_data = calibration_data.loc[(calibration_data["A"] != 0) & (df["A"] != 1)]
    model = RealWorldRegressor.load_from_checkpoint('models/Regression_mimic/Regression_mimic_checkpoints.ckpt').eval()
    propensity_estimator = CondNormalizingFlow.load_from_checkpoint('models/Density_estimator_mimic_checkpoints.ckpt').eval()

    for alpha in evaluation_alphas:

        #mc_indices = np.where(alpha_list == alpha)[0]
        for ind in test_data.index:
            # CP-bounds
            sample = test_data[ind: ind+1]
            model.eval()
            optim = prediction_intervals(model = model.eval(), calibration_data=calibration_data, new_sample=sample, intervention_mode="unknown", alpha=alpha, eps=eps, prop_estimator = propensity_estimator, M = M, task = "mimic")#, conf_scores = conf_scores)
            l, u = optim.create_interval()
            # MC-bounds
           #mc_quantiles = model.predict_step(torch.Tensor(test_data.loc[[ind]].drop("Y", axis=1).values))

            model.eval()
            small_df = pd.DataFrame({"a_star": sample["A"], "alpha": [alpha], "eps": [eps], "M": [M], "gender": sample["gender"], "age":sample["age"], "heart rate": sample["heart rate"], "sodium":sample["sodium"], "blood pressure":sample["blood pressure"], "glucose":sample["glucose"], "hematocrit": sample["hematocrit"], "respiratory rate":sample["respiratory rate"], "A": sample["A"].values, "Y": sample["Y"], "Y_pred": model(torch.Tensor(sample.loc[:,sample.columns != "Y"].values)).detach().numpy()[0], "C_low": l, "C_up": u, "C_low_mc": 0, "C_up_mc":0})#[mc_quantiles[mc_indices[0]]], "C_up_mc": [mc_quantiles[mc_indices[1]]]})
            df_intervals = pd.concat([df_intervals, small_df])
            print(l, u)

            df_intervals.to_csv('results/prediction_intervals_mimic_semi.csv', index=False) 





def generate_df_unknown_prop(dataset_list, intervention_list, evaluation_alphas, df_path):

    '''
        First need to have generated an empty dataframe via 
        df = pd.DataFrame(columns=["dataset", "a_star", "alpha", "eps", "M", "X", "A", "Y", "Y_pred", "C_low", "C_up", "C_low_mc", "C_up_mc"])
        and saved it at
        df.to_csv('results/prediction_intervals_simulated_unknown.csv', index=False) 
    '''
    alpha_list = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 0.5, 0.2, 0.1, 0.05, 0.01])        # MC- quantiles outputed by prediction step in regression model
    eps = 0.0001
    M = 2

    df_intervals = pd.read_csv(df_path) 

    time_list = []
    for dataset in dataset_list:   

        calibration_data = pd.read_csv('data/simulated_data' + str(dataset) + '_calibration.csv') 
        model = SimpleRegressionModel.load_from_checkpoint('models/Regression_simulated_' + str(dataset) + '_checkpoints.ckpt').eval()
        #conf_scores = conformity_score(y = np.reshape(calibration_data["Y"], (-1,1)), y_hat = model(torch.Tensor(calibration_data[["X", "A"]].values)).detach().numpy())
        #print(conf_scores.squeeze())
        propensity_estimator = CondNormalizingFlow.load_from_checkpoint('models/Density_estimator' + str(dataset) + '_checkpoints.ckpt').eval()

        for a_star in intervention_list:
            print(a_star)

            data = pd.read_csv('data/simulated_data' + str(dataset) + '_Intervention' + str(a_star) + '.csv')
            data = data.loc[data["A"] <= 40]
            data_X1 = data.loc[data["X"]==1][0:1]#[0:40]
            data_X2 = data.loc[data["X"]==2][0:1]
            data_X3 = data.loc[data["X"]==3][0:1]
            data_X4 = data.loc[data["X"]==4][0:1]
            data = pd.concat([data_X1, data_X2], ignore_index=True)
            data = pd.concat([data,data_X3], ignore_index=True)
            data = pd.concat([data,data_X4], ignore_index=True)

            for alpha in evaluation_alphas:
                print(alpha)

                mc_indices = np.where(alpha_list == alpha)[0]
                for ind in data.index:
                    # CP-bounds
                    sample = data.loc[[ind]]
                    model.eval()
                    optim = prediction_intervals(model = model.eval(), calibration_data=calibration_data, new_sample=sample, intervention_mode="unknown", intervention=a_star, alpha=alpha, eps=eps, prop_estimator = propensity_estimator, M = M, task = dataset)
                    
                    start = timeit.default_timer()
                    l, u = optim.create_interval()
                    stop = timeit.default_timer()
                    time_list += [stop-start]

                    # MC-bounds
                    mc_quantiles = model.predict_step(torch.Tensor(data.loc[[ind]].drop("Y", axis=1).values))

                    model.eval()
                    small_df = pd.DataFrame({"dataset": [dataset], "a_star": [a_star], "alpha": [alpha], "eps": [eps], "M": [M], "X": sample["X"].values, "A": sample["A"].values, "Y": sample["Y"].values, "Y_pred": model(torch.Tensor(sample[["X", "A"]].values)).detach().numpy()[0], "C_low": [l], "C_up": [u], "C_low_mc": [mc_quantiles[mc_indices[0]]], "C_up_mc": [mc_quantiles[mc_indices[1]]]})
                    df_intervals = pd.concat([df_intervals, small_df])
                    print(l, u)

                    df_intervals.to_csv(df_path, index=False) 
    print(time_list)          
    print(np.mean(time_list))                
        




# Known intervention/propensity

def generate_df_known_prop(dataset_list, delta_list, covariates_list, evaluation_alphas, df_path):

    '''
        First need to have generated an empty dataframe via 
        df = pd.DataFrame(columns=["dataset", "delta", "alpha", "eps", "X", "A", "Y", "Y_pred", "C_low", "C_up", "C_low_mc", "C_up_mc"])
        and saved it at
        df.to_csv('results/prediction_intervals_simulated_known.csv', index=False) 
    '''

    alpha_list = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 0.5, 0.2, 0.1, 0.05, 0.01])        # MC- quantiles outputed by prediction step in regression model
    eps = 0.1

    df_intervals = pd.read_csv(df_path) 
    time_list = []
    for dataset in dataset_list:  

        calibration_data = pd.read_csv('data/simulated_data' + str(dataset) + '_calibration.csv') 
        model = SimpleRegressionModel.load_from_checkpoint('models/Regression_simulated_' + str(dataset) + '_checkpoints.ckpt').eval()

        for delta in delta_list:
            for cov in covariates_list:

                data = pd.read_csv('data/simulated_data' + str(dataset) + '_Delta' + str(delta) + '_' + cov + '.csv').sort_values(by="A", axis = 0)
                data = data.loc[data["A"] <= 40]
                for alpha in evaluation_alphas:

                    mc_indices = np.where(alpha_list == alpha)[0]
                    for ind in data.index:
                        # CP-bounds
                        sample = data.loc[[ind]]
                        model.eval()

                        start = timeit.default_timer()
                        optim = prediction_intervals(model = model, calibration_data=calibration_data, new_sample=sample, intervention_mode="known", alpha=alpha, eps=eps, delta_A= delta, task = dataset)
                        stop = timeit.default_timer()
                        time_list += [stop-start]

                        l, u = optim.create_interval()
                        # MC-bounds
                        mc_quantiles = model.predict_step(torch.Tensor(data.loc[[ind]].drop("Y", axis=1).values))

                        model.eval()
                        small_df = pd.DataFrame({"dataset": [dataset], "delta": [delta], "alpha": [alpha], "eps": [eps], "X": sample["X"].values, "A": sample["A"].values, "Y": sample["Y"].values, "Y_pred": model(torch.Tensor(sample[["X", "A"]].values)).detach().numpy()[0], "C_low": [l], "C_up": [u], "C_low_mc": [mc_quantiles[mc_indices[0]]], "C_up_mc": [mc_quantiles[mc_indices[1]]]})
                        df_intervals = pd.concat([df_intervals, small_df])

                df_intervals.to_csv(df_path, index=False) 
    print(time_list)          
    print(np.mean(time_list))  



def intervals_known_prop(seed_list, dataset_list, delta_list, covariates_list, evaluation_alphas, df_path):
    
    alpha_list = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 0.5, 0.2, 0.1, 0.05, 0.01])        # MC- quantiles outputed by prediction step in regression model
    eps = 0.1
    df_all = pd.read_csv(df_path) 

    for dataset in dataset_list:  

        calibration_data = pd.read_csv('data/simulated_data' + str(dataset) + '_calibration.csv') 

        for seed in seed_list: 
            if seed == 2024:
                model = SimpleRegressionModel.load_from_checkpoint('models/Regression_simulated_' + str(dataset) +  '_checkpoints.ckpt').eval()
            else:
                model = SimpleRegressionModel.load_from_checkpoint('models/Regression_simulated_' + str(dataset) + '_seed'+ str(seed) +  '_checkpoints.ckpt').eval()

            np.random.seed(2024) 
            pl.seed_everything(2024)

            for delta in delta_list:
                for cov in covariates_list:

                    data = pd.read_csv('data/simulated_data' + str(dataset) + '_Delta' + str(delta) + '_' + cov + '.csv').sort_values(by="A", axis = 0)
                    data = data.loc[data["A"] <= 40]
                    for alpha in evaluation_alphas:

                        mc_indices = np.where(alpha_list == alpha)[0]
                        for ind in data.index:
                            # CP-bounds
                            sample = data.loc[[ind]]
                            model.eval()
                            optim = prediction_intervals(model = model, calibration_data=calibration_data, new_sample=sample, intervention_mode="known", alpha=alpha, eps=eps, delta_A= delta, task = dataset, seed = seed)
                            l, u = optim.create_interval()
                            # MC-bounds
                            mc_quantiles = model.predict_step(torch.Tensor(data.loc[[ind]].drop("Y", axis=1).values))

                            if ((sample["Y"].values <= u) & (sample["Y"].values >=l)):
                                coverage = 1
                            else:
                                coverage = 0   

                            if (sample["Y"].values <= mc_quantiles[mc_indices[1]]) & (sample["Y"].values >=mc_quantiles[mc_indices[0]]):
                                coverage_mc = 1
                            else:
                                coverage_mc = 0                                 


                            model.eval()
                            small_df = pd.DataFrame({"seed": [seed], "dataset": [dataset], "delta": [delta], "alpha": [alpha], "coverage": [coverage], "coverage_mc": [coverage_mc],"length": [u-l], "length_mc": [mc_quantiles[mc_indices[1]] - mc_quantiles[mc_indices[0]]]})
                            df_all = pd.concat([df_all, small_df])
                    
                    df_all.to_csv(df_path, index=False) 




