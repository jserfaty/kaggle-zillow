# TODO:
# Try removing outliers
# Remove features / examples with lots of missing data
# Look at time series trends to adjust estimates over time
# Additional feature engineering:
#   Infer neighborhoods
#   Distance from coast
#   Add in more binary / one-hot encoded variables
#   Transform single variables w/ xgboost and feed back in as features
# Stack models

import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import KFold
import xgboost as xgb

def expand_grid(data_dict):
    """Create a dataframe from every combination of given values."""
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())

def clean_data(data_frame):
    """Clean dataframe: change float64 to 32 and change object to bool for xgboost"""
    for c, dtype in zip(data_frame.columns, data_frame.dtypes):	
        if dtype == np.float64:		
            data_frame[c] = data_frame[c].astype(np.float32)

    for c in data_frame.dtypes[data_frame.dtypes == object].index.values:
        data_frame[c] = (data_frame[c] == True)

    return(data_frame)

# Load data
train = pd.read_csv('data/train_2016.csv')
prop = pd.read_csv('data/properties_2016.csv')

# Clean features
prop = clean_data(prop)

# Join features with labels
train_df = train.merge(prop, how='left', on='parcelid')

# Refine training data: Remove columns with too much missing info / not useful at the moment; split out labels
x_df = train_df.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
y_df = train_df['logerror'].values
columns_df = x_df.columns

# Convert to np array for xgboost
x_df_np = x_df.values.astype(np.float32,copy=False)

# Create KFold splitter
N_SPLITS = 10
fold_idx = KFold(n_splits=N_SPLITS,random_state=0,shuffle=True)
fold_idx.get_n_splits(x_df)

# Define parameters for grid search
params = {}
params['max_depth'] = [2,4,6,8,10]              # Max depth of tree (smaller = less overfitting)
params['eta'] = [0.01,0.1,0.3,1]                # Learning rate (smaller = shrinks feature weights to be more conservative)
params['gamma'] = [0,1,10,100]                  # Min split loss required for additional split (larger = less overfitting)
params['lambda'] = [0,0.5,1]                    # L2 regularization term (larger = less overfitting)
params['alpha'] = [0,0.5,1]                     # L1 regularization term (larger = less overfitting)
params['subsample'] = [0.33,0.66,1]             # Fraction of examples used to grow trees (smaller = less overfitting)
params['colsample_bytree'] = [0.33,0.66,1]      # Fraction of features used to grow trees (smaller = less overfitting)
params['nround'] = [1000]                       # Number of rounds for boosting
params['early_stopping_round'] = [10]           # Stop if no improvement in metric
params['eval_metric'] = ['rmse']
params['objective'] = ['reg:linear']
param_grid = expand_grid(params)

# Initialize dataframe for holding grid search CV results
scores = pd.DataFrame(data=None, columns=['trial','error','param_set_num','max_depth','eta','gamma','lambda','alpha'
                                          ,'subsample','colsample_bytree','nround','early_stopping_round','eval_metric','objective'])

# Run through Kfold cv on parameter grid
trial = 0
for index,param_set in param_grid.iterrows():
    param_set_num = index
    max_depth = param_set['max_depth']
    eta = param_set['eta']
    gamma = param_set['gamma']
    lambda_ = param_set['lambda']
    alpha = param_set['alpha']
    subsample = param_set['subsample']
    colsample_bytree = param_set['colsample_bytree']
    nround = param_set['nround']
    early_stopping_round = param_set['early_stopping_round']
    eval_metric = param_set['eval_metric']
    objective = param_set['objective']

    # For each parameter set, run on K folds and save errors on validation set
    for train_index, test_index in fold_idx.split(x_df):
        X_train, X_test = x_df_np[train_index], x_df_np[test_index]
        y_train, y_test = y_df[train_index], y_df[test_index]

        d_train = xgb.DMatrix(X_train, label = y_train)
        d_test = xgb.DMatrix(X_test, label = y_test)

        xgb_model = xgb.train(dtrain = d_train, params = {'max_depth':max_depth, 'eta':eta, 'gamma':gamma, 'lambda':lambda_, 'alpha':alpha
                                                          ,'subsample':subsample,'colsample_bytree':colsample_bytree
                                                          ,'nround':nround,'early_stopping_round':early_stopping_round
                                                          ,'eval_metric':eval_metric,'objective':objective})
        xgb_pred = xgb_model.predict(d_test)

        error = mean_squared_error(y_test, xgb_pred)

        scores.loc[trial] = [trial,error,param_set_num,max_depth,eta,gamma,lambda_,alpha,subsample,colsample_bytree
                             ,nround,early_stopping_round,eval_metric,objective]

        trial += 1
        print(str(trial) + " of " + str(param_grid.shape[0]))

# Save scores for reference
scores.to_csv('xgb_model3_scores.csv', index=False, float_format='%.6f')

# Find average error over the folds for each parameter set 
avg_error = scores.groupby('param_set_num').mean().sort('error').reset_index()

# Get optimal parameters for final model
param_opt = {'max_depth': avg_error['max_depth'][0]
             ,'eta': avg_error['eta'][0]
             ,'gamma': avg_error['gamma'][0]
             ,'lambda': avg_error['lambda'][0]
             ,'alpha': avg_error['alpha'][0]
             ,'subsample': avg_error['subsample'][0]
             ,'colsample_bytree': avg_error['colsample_bytree'][0]
             ,'nround': avg_error['nround'][0]
             ,'early_stopping_round': avg_error['early_stopping_round'][0]
             }
print(param_opt)

# Build model using optimal parameters
xgb_final = xgb.train(dtrain = d_train, params = {'max_depth':int(param_opt['max_depth'])
                                                  ,'eta':param_opt['eta']
                                                  ,'gamma':param_opt['gamma']
                                                  ,'lambda':param_opt['lambda']
                                                  ,'alpha':param_opt['alpha']
                                                  ,'subsample':param_opt['subsample']
                                                  ,'colsample_bytree':param_opt['colsample_bytree']
                                                  ,'nround':param_opt['nround']
                                                  ,'early_stopping_round':param_opt['early_stopping_round']
                                                  ,'objective':objective })

# Load sample submission to get submission parcelids
sample = pd.read_csv('data/sample_submission.csv')

# Copy submission to fill in later with actual predicions
sub = sample.copy()

# Join property data to submission sample
sample['parcelid'] = sample['ParcelId']
sample_data = sample.merge(prop, on='parcelid', how='left')

# Create test data using only necessary columns
X_sample = sample_data[columns_df]

# Convert test data into format for xgboost
for c in X_sample.dtypes[X_sample.dtypes == object].index.values:
    X_sample[c] = (X_sample[c] == True)
X_sample_np = X_sample.values.astype(np.float32,copy=False)
d_sample = xgb.DMatrix(X_sample_np)

# Use final model to predict outcome on test data
y_pred = xgb_final.predict(d_sample)

# Fill in submission with predictions
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = y_pred

# Write submissions to csv
sub.to_csv('xgb_submission_3.csv', index=False, float_format='%.4f')