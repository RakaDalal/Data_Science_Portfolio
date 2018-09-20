# House-Price-Prediction

I submitted this project to a Kaggle competition.

The dataset comprised of 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa. The training data has 1461 rows of 81 variables. 

The aim is to predict the final price of each home. 

I explored the data and performed visualization. I did extensive feature engineering to select important features and also created new features. I have used PCA to further reduce the dimensionality. Finally, I have used models like linear regression, ridge regression, decision tree regressor, random forest regressor, XGB regressor. The XGB regressor outperformed former models. So, I tuned the parameters of XGB regressor to further improve the results. Next I used a neural network with two hidden layers which performed best among all models. Each hidden layer is comprised of 256 hidden nodes. I have used a batch size of 64 and number_of_epochs=800.
