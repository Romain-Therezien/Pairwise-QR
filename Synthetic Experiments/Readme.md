# Synthetic Experiments for Quantile Regression
This directory contains code for synthetic experiments on quantile regression. The experiments are designed to evaluate the performance of different quantile regression methods under various conditions. 
The code is available in main.py, which includes functions for generating synthetic data, fitting quantile regression models, and evaluating their performance.
The synthetic data is generated using a specified distribution, and the quantile regression models are fitted to the data to estimate the conditional quantiles. The performance of the models is evaluated using metrics such as the pinball loss and the coverage of the predicted quantiles. 
The code is structured to allow for easy modification and experimentation with different data generation processes, quantile regression methods, and evaluation metrics. 

The hyperparameters grid search is implemented in Grid Search Boosting/LightGBM.py, which allows for tuning the parameters of the Gradient Boosting and LightGBM model used in the quantile regression experiments. The grid search is performed over a specified range of hyperparameters, and the best combination is selected based on the evaluation metrics.

