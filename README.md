# Titanic submission
I tried two different models, the first a neural network. The second a meta ML model.
## Neural Network
I first tried to use a neural network, with two hidden layers with 16 and then 12 neurons, with a dropout of 0.5, reaching an accuracy of only 0.55. With some slight optimisation of the neural network's layers, having the first hidden layer have 64 units and using a relu activation function for the middle layers and a sigmoid function for the binary end ouput the accuracy increased to around 0.75.


### Feature engineering
I replaced any na's in the age (there were 177) with the mean, increasing the accuracy to 0.8 on the cross validation data-set.

## Stacking models
With just the neural network I plateaud at 80% accuracy, so I tried stacking several models, using a meta linear regression model which took inputs from the following models.

* Logistic Regression 
  * Accuracy on the cross validation test set: 0.80
  * Hyperparameters: Used the liblinear solver as this is a relatively small dataset.
* Random Forests
  * Accuracy on the cross validation test set: 0.84
  * Explanation: This is a learning algorithm that creates trees of if else statements, with leaf values at the end of every node, these are usually based on the average of the rest of the data points that have the same features.
  * Sources: https://www.kaggle.com/dansbecker/random-forests
* K neighbours classifier
  * Accuracy on the cross validation test set: 0.82 (using k=8, tried 3,5,8 and 10)
  * Hyperparameters: {'weights': 'uniform', 'p': 1, 'n_neighbors': 8}
  * Explanation: This plots all the datapoints and makes a prediction on an unkown datapoint based on its k nearest neighbours.
* Gaussian Naive bayes classifier
  * Accuracy on the cross validation test set: 0.8
  * Explanation: Uses conditional probability (mainly bayes' theorem) to get the probability of the outcome, using previous data to calculate the probaility values. A Gaussian classifier then assumes the inputs follow a gaussian distribution, which is true for most real world data, such as the age or fare. Making it easier to calculate the probabilities.
  * More info: https://www.geeksforgeeks.org/naive-bayes-classifiers/
* SVC
  * Accuracy on the cross validation test set: 0.81
  * Explanation: Creates margins in multi dimensional data that tries to seperate data.
* Gradient boosting classifier & xgbc classifier
  * More info: https://blog.paperspace.com/gradient-boosting-for-classification/

## Stacking the models - method
The first model had only the Gaussian NB, logistic regression and random forest classifier as input models. This lead to a mean accuracy of: 0.82. With the hyperparameters optimised it went to: 0.824, in short hyperparameter optimisation matters very little, the defaults are usually good enough for a quick project.

I then added had a model with all the above models as the input models. The model then reached an accuracy of 0.82. However, this was on the real test set, so it can be assumed that it performs slightly better than the single models alone. However, the model takes several hours to train so it was probably not worth it.

### Explanation of evaluate_model

Uses k-fold cross-validation with stratified picking of the groups. This negates the imbalance of the two outcomes, more people died than survived as ensures that in each group picked there. However, this is mostly best practice, the difference isn't that big and the test set is relatively large.

Refrence: https://machinelearningmastery.com/cross-validation-for-imbalanced-classification/, https://machinelearningmastery.com/k-fold-cross-validation/

## Hyper parameter optimization
For Logistic regression used grid search CV to get the optimal hyperparameters which were C = 0.1, penalty = l2 and solver = newton-cg. Gave accuracy of 0.80.
More info: https://botbark.com/2019/12/25/top-5-hyper-parameters-for-logistic-regression/

For Random tree classifier used ```random search CV``` to get good hyperparams, ```grid search``` took too long. After 20 iterations the best params were: {'n_estimators': 50, 'min_samples_split': 6, 'max_features': 'auto', 'max_depth': 10}. With accuracy of 0.83. More info: https://www.geeksforgeeks.org/hyperparameters-of-random-forest-classifier/ 

For Gaussian naive bayes used grid search as not that many params. The best params were: var_smoothing = 0.35 with accuracy 0.79.
Var smoothing is the value to add to the gaussian curve generated based on the mean and std of every feature. A positive value widens or smoothes the curve, therefore the curve is more likely to contain outliers delimiting the boundary, so will decrease the specificity but increase the sensitivity. here it is trained on accuracy.

## Acknowledgments
Much of the stacking ML models code comes from here: https://www.kaggle.com/rushikeshdarge/competition-winning-models-stacking
The feature engineering ideas came from here: https://www.kaggle.com/omarelgabry/a-journey-through-titanic
