import pickle
import gzip
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_score

path_to_model="./"
f = gzip.open(path_to_model+'modified_data.pklz','rb')
modified_data = pickle.load(f)
f.close()
print ("reading done")

perceptron = Perceptron()

#train model with cv of 5 
cv_scores = cross_val_score(perceptron, modified_data["X_train"], modified_data["Y_train"], cv=5)

print(‘cv_scores mean:{}’.format(np.mean(cv_scores)))