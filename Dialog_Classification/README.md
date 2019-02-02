# Dialog_classification

The packages required is provided in the file requirements.txt. In order to install the packages run

pip install -r requirements.txt


The environment used is Python 2. 

I have solved Machine Learning Programming problem which involves building a classifier 
that is able to predict the character that has spoken a dialogue. 

There exists two coding files:
1.	Feature_engineering_and training.ipynb
2.	prediction.py

In order to serialize the pipeline, the training model is saved in the following two files:
1. dictionary.pklz
2. my_trained_model.h5

The file simpsons_dataset.csv should be in the same folder to run the Jupiter notebook file.

Feature Engineering and Training

The jupyter notebook file (feature_engineering_and_training.ipynb) contains the code for 
data-preprocessing (handling imbalanced and missing data), feature engineering (linguistic and statistical) and spot checking the performance of different Machine Learning classifiers (via training and validation split in the ratio of 4:1) that include Naive Bayes classifier, Logistic Regression, Artificial Neural Network (with 2 hidden layers), Convolutional Neural Network and Bidirectional LSTM. I showed that the model that works best over the validation data is the Artificial Neural Network and therefore that was chosen as our best model. Subsequently the ANN was trained over the entire data and was saved in order to serialize. The serialized model has also been provided along with this exercise so that running the jupyter notebook is not essential.

Prediction

How to run: 

python prediction.py <input_text>
For example, python prediction.py "Hey, thanks for your vote, man"

This code takes as input a dialogue entry and outputs one of the five values: Homer Simpson, Marge Simpson, Bart Simpson, Lisa Simpson, or Other.

Problem_2.pdf contains the suggested improvements to the model.
