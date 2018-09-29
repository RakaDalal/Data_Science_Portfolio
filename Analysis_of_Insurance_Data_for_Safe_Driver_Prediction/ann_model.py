from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
import pickle
import gzip

path_to_model="./"
f = gzip.open(path_to_model+'modified_data.pklz','rb')
modified_data = pickle.load(f)
f.close()
print ("reading done")

def build_classifier(optimizer, hidden_nodes):
    classifier = Sequential()
    classifier.add(Dense(units = hidden_nodes, kernel_initializer = 'uniform', activation = 'relu', input_dim = 47))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units = hidden_nodes, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units = hidden_nodes, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [32, 64, 128],
              'epochs': [500, 800],
              'hidden_nodes': [128, 256],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5)
grid_search = grid_search.fit(modified_data["X_train"], modified_data["Y_train"])
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print (best_parameters)
print (best_accuracy)