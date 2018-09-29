import pickle
import gzip
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

path_to_model="./"
f = gzip.open(path_to_model+'modified_data.pklz','rb')
modified_data = pickle.load(f)
f.close()
print ("reading done")

# Tuning of parameters for XGBClassifier
# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [30, 40, 50],
    'n_estimators': [100, 500, 800],
    'learning_rate': [0.05, 0.1, 0.2]
}
# Create a based model
gbm = xgb.XGBClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = gbm, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)
# Fit the grid search to the data
grid_search.fit(modified_data["X_train"], modified_data["Y_train"])
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print (best_parameters)
print (best_accuracy)