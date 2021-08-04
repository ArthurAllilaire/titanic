# Titanic submission
## Method
Used a neural network to try and predict the survivors.

## Different parameters tried

### Initial setup
Had two hidden layers with 16 and then 12 neurons, with a dropout of 0.5. No feature engineering. the neural network reached an accuracy of 0.613 after 100 epochs. However, this used the passenger Id, without the id the accuracy is 0.55.

Got rid of the na's in the age (there were 177) by replacing with the mean, increased accuracy to 0.8 on the cross validation.

### Feature engineering

### Activation functions
Used relu for middle layers, then sigmoid for the binary end output - need to learn what they are when asked in presentation.

### Hidden layers
Got rid of the 16 unit hidden layer, led to a slight increase in accuracy to 0.644 on the validation test set (which is 10% of training test set). 

Changed the first hidden layer from 12 to 64 units, increased accuracy to 0.86, this was mostly luck, the accuracy plateaus at around 80% without any feature engineering.

### Dropout
Once I increased the layers to 64 and 12 increased dropout to 0.5 to reduce overfitting.


## Stacking models
With just the neural network I plateaud at 80% accuracy, so I tried stackin models, using a meta model to choose based on the individual models.

### Models
#### Logistic Regression
Used logistic regression and got an accuracy of 0.80 by itself. used the liblinear solver as this is a relatively small dataset.

#### Random Forests
This is a learning algorithm that creates trees of if else statements, with leaf values at the end of every node, these are usually based on the average of the rest of the data points that have the same features. Implemented this and got an accuracy of 0.84 on the validation test set.
More info: https://www.kaggle.com/dansbecker/random-forests

#### K neighbours classifier
This plots all the datapoints and makes a prediction on an unkown datapoint based on its k nearest neighbours. The accuracy with 5 nearest neighbours is 0.8044692737430168, tried with 3,5,8 and 10, best was 8 with accuraccy of 0.82.

#### Gaussian NB
Info: https://www.geeksforgeeks.org/naive-bayes-classifiers/

Has an accuracy of 0.80.
##### Naive bayes classifier explanation
Uses conditional probability (mainly bayes' theorem) to get the probability of the outcome, using previous data to get the probailities

A Gaussian classifier then assumes the inputs follow a gaussian distribution, which is true for most real world data, such as age or fare. Makes it easier to calculate the probabilities.

#### SVC
Creates margins in multi dimensional data that tries to seperate data. It had an accuracy of: 0.81.

## Stacking the models
Use a linear regression ML model to take input from all the other ML models and make the final prediciton.

The first stacking method, with model 0's of Gaussian NB, logistic regression and random forest classifier had an accuracy of: 0.82. That is the mean over all tries. With the hyperparameters optimised it went to: 0.824, in short hyperparameter optimisation matters very little, the defaults are usually good enough for a quick project.

For now I don't know how to add the neural network.
I then added my neural network to the estimators, with untested hyperparameters.

Instead added a gradient boosting classifier and xgbc classifier. The first works by 

Source: https://blog.paperspace.com/gradient-boosting-for-classification/

Also added a k-nearest-neighbours classifier, had accuracy of 0.82 with hyperparams = {'weights': 'uniform', 'p': 1, 'n_neighbors': 8} on the training data by itself. The overall model has an accuracy of 0.82.

Used logistic regression for final_estimator, took too long to try other models, the accuracy overall was 

### Meta learner experimenation

Sources: https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/

### Explanation of evaluate_model

Uses k-fold cross-validation with stratified picking of the groups. This negates the imbalance of the two outcomes, more people died than survived as ensures that in each group picked there. However, this is mostly best practice, the difference isn't that big and the test set is relatively large.

Refrence: https://machinelearningmastery.com/cross-validation-for-imbalanced-classification/, https://machinelearningmastery.com/k-fold-cross-validation/

## Hyper parameter optimization
For Logistic regression used grid search CV to get the optimal hyperparameters which were C = 0.1, penalty = l2 and solver = newton-cg. Gave accuracy of 0.80.
More info: https://botbark.com/2019/12/25/top-5-hyper-parameters-for-logistic-regression/

For Random tree classifier used random search CV to get good hyperparams, grid search took too long. After 20 iters the best params were: {'n_estimators': 50, 'min_samples_split': 6, 'max_features': 'auto', 'max_depth': 10}. With accuracy of 0.83. more info: https://www.geeksforgeeks.org/hyperparameters-of-random-forest-classifier/ 

For Gaussian naive bayes used grid search as not that many params. The best params were: var_smoothing = 0.35 with accuracy 0.79.
Var smoothing is the value to add to the gaussian curve generated based on the mean and std of every feature. A positive value widens or smoothes the curve, therefore the curve is more likely to contain outliers delimiting the boundary, so will decrease the specificity but increase the sensitivity. here it is trained on accuracy.

For the neural network I tried 

## Acknowledgments
Much of the stacking ML models code comes from here: https://www.kaggle.com/rushikeshdarge/competition-winning-models-stacking
The feature engineering ideas came from here: https://www.kaggle.com/omarelgabry/a-journey-through-titanic

# titanic
