Churn Prediction
====================

In this problem, we are asked to analyze the past data of a customer and predict whether the customer will churn or not in the future.
--------------------------------------------------------------------------------------------------------------------------------------

First of all, I completed several exploratory data analysis tasks and subsequently pre-processed the data:

1) Imbalanced Data: The data was imbalanced i.e. the number of customers who
churned was significantly smaller than the number of customers who did not.
There are multiple ways of handling data imbalance:
  a) Oversampling: Since the dataset was small, oversampling the positive
class led to many repetitions of the positive data points. As a result, the
model was overfitting to those repeated data points while training; this led
to a high training macro f1 score but a poor average macro f1 score during
cross-validation.
  b) Tuning Class Weights: We can tune the class weights provided in
sklearn ML modules as a hyperparameter. This technique with the final
class weights of {0:0.3, 1:0.7} provided the best results for handling
imbalanced data.

2) Age: This is arguably the most important feature since it has the highest
correlation with the output. To allow for interpretability and better analysis, the
Age feature (continuous) was discretized. In this operation, we experimented with
the number of bins as a hyperparameter and found the best results with 10 bins
each of uniform size.

3) Balance: The Balance feature takes a wide range of values and therefore
discretizing this is not a good idea. In contrast, we take the log of this feature to
squeeze the values in the feature into a much smaller range; another advantage
of taking log is that the features look more like a Gaussian.

4) Outlier detection based on Balance Feature: As described before, the
Balance feature takes a few extreme values in some data points. One way to
handle this is to remove those data points from the training data; we remove
those data points where the balance value is 2 standard deviations away from
the mean. However, this approach led to a decrease in the average
cross-validated macro F1 score. This is because the dataset is small and
throwing away data points is not a good approach for this problem. Again, the log
transformation comes to our rescue.

5) New features:
  a ) Since Age had the highest correlation with the output, we created
several new features that are non-linear transformations namely elementwise
powers of the Age feature. This is particularly helpful since we found that Logistic
Regression (which learns a linear separator between the classes) was the best
performing model (more on this later). We used up to the 6th power of the Age
feature (adding more led to overfitting and a drop in the average cross-validated
macro F1 score)
  b) We also used the element-wise product of the Age and Transaction
Status feature (with 0’s replaced by -1’s) and similarly the element-wise product
of the Age and the Gender feature (with Females replaced by -1 and Males
replaced by +1 ).
  c) We created 4 more features which are combinations of the Transaction
feature and the Gender feature i.e. the Transaction and the Gender features are
both binary and there are 4 possibilities that each pair of them can take (0,0),
(0,1), (1,0) and (1,1); We created a one-hot encoding of this set of 4 possibilities
i.e. each one-hot encoded feature is 1 when only one particular possibility is true
and is 0 otherwise.

6) Remove unimportant features: We drop the features Credit_card, Income,
Vintage since they have a small correlation with the output. Keeping them leads
to overfitting.

Model training and Hyper-parameter tuning
------------------------------------------

Next, we move on to model training and hyper-parameter tuning. We tried several base
models including Logistic Regression (with L1 and L2 penalty), LDA, Perceptron, SVM,
Decision Tree, Random Forest, XGBoost, k-Nearest Neighbors and Naive Bayes with
10-fold Cross-Validation to understand the generalization properties. Among these,
logistic regression with L1 penalty had the best average cross-validation macro f1
score and therefore, we chose this model for final submissions. This is because logistic
regression with L1 penalty is also performing an implicit feature selection which avoids
overfitting in the setting when the number of features is large. We further tuned the
parameters of this model including regularization parameter and classification threshold
using Cross-Validation for best results.

Summarization
--------------

In summary, our best submission was by training a Logistic regression model with L1
penalty C=0.95 and class weights {0:0.3, 1:0.7} along with the feature engineering steps
described above.
