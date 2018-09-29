import pickle
import gzip
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

path_to_model="./"
f = gzip.open(path_to_model+'modified_data.pklz','rb')
modified_data = pickle.load(f)
f.close()
print ("reading done")

# Tuning of parameters for Random Forest
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [20, 40, 50],
    'max_features': [6, 8, 10, 12],
    'n_estimators': [200, 500, 800]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)
# Fit the grid search to the data
grid_search.fit(modified_data["X_train"], modified_data["Y_train"])
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print (best_parameters)
print (best_accuracy)