# Analysis of Insurance Data for Safe Driver Prediction

I submitted this project to a Kaggle competition.

The aim of the project is to build a model that predicts the probability that a driver will initiate an auto insurance claim in the next year. The dataset (Porto Seguro dataset) comprised of 57 features. The training data has 595212 rows of 59 variables.

I explored the data and performed visualization. One major problem was that the dataset was imbalanced. As the training set was huge, I performed undersampling to deal with the problem. Next, I did extensive feature engineering to select important features. I handled missing values. I used a correlation matrix to determine the correlation among different features and used logistic regression to determine the importance of different features. Finally, after feature scaling I have used models like logistic regression, k-nearest neighbour, perceptron, gaussian Naive Bayes, decision tree, random forest, XGB classifier and artificial neural network. I have also tuned the parameters of random forest, XGB classifier and artificial neural network to get better results. The tuned artificial neural network outperformed all other models in cross-validation. I used three hidden layers with 256 hidden nodes in each. I used dropout in each layer. I have used a batch size of 64 and number_of_epochs=800. 
