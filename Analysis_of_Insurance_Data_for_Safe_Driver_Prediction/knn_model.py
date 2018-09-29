import pickle
import gzip
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

path_to_model="./"
f = gzip.open(path_to_model+'modified_data.pklz','rb')
modified_data = pickle.load(f)
f.close()
print ("reading done")

knn = KNeighborsClassifier(n_neighbors = 4)

#train model with cv of 5 
cv_scores = cross_val_score(knn, modified_data["X_train"], modified_data["Y_train"], cv=5)

print(‘cv_scores mean:{}’.format(np.mean(cv_scores)))