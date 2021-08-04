from math import nan
import numpy as np  # linear algebra

# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import csv

# Neural network imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu, elu


# Other models used
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# Evaluation imports
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.model_selection import cross_val_score

# Imports for stacking the models together
from sklearn.ensemble import StackingClassifier

EPOCHS = 10


def load_data():
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")
    # Name is irrelevant
    features = ['Pclass', 'Sex', 'Age', 'SibSp',
                'Parch', 'Fare', 'Embarked']
    # Get labels
    y = train_data["Survived"]

    X = train_data[features]
    X_test = test_data[features]

    return X, X_test, y


def clean_data(X, X_test):
    """
    Gets rid of the nan values
    returns nothing but modifies the original X and X_test
    """
    print(X.isnull().sum())
    # Change the AGE nan values
    # replace the nan with the mean
    X["Age"].fillna(value=X["Age"].mean(), inplace=True)
    # replace the nan with the mean
    X_test["Age"].fillna(value=X_test["Age"].mean(), inplace=True)

    # Change the Fare nan value (there is only 1)
    X_test["Fare"].fillna(X_test["Fare"].mean(), inplace=True)

    # only in titanic_df, fill the two missing values with the most occurred value, which is "S".
    X["Embarked"] = X["Embarked"].fillna("S")

    print(X_test.isnull().sum())

    return X, X_test


def prepare_data(split=True):
    X, X_test, y = load_data()

    # Get rid of nan values
    clean_data(X, X_test)

    # Convert pd to numpy arrays and make everything to numbers
    X = pd.get_dummies(X).values
    X_test = pd.get_dummies(X_test).values

    # Normalise the data
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X_test = sc.fit_transform(X_test)

    if split:
        # Create a validation set for testing the neural network
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=0)

        return X_train, X_val, y_train, y_val, X_test, y

    return X, X_test, y


# def get_logistic_regression_model(feature_num):


def predict_labels(model, X_test, filename="submission.csv"):
    """
    Generates predictions and writes them to file
    """
    threshold = 0.5
    y_pred = model.predict(X_test)

    # convert the probabilities to binary based on threshold
    y_pred = [int(x > threshold) for x in y_pred]

    # Get the passenger id's
    passengerId = pd.read_csv("test.csv")["PassengerId"]

    # Write to the CSV file
    header = ['PassengerId', 'Survived']

    # So no extra \n below header
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(header)

        for i in range(len(passengerId)):
            writer.writerow([passengerId[i], y_pred[i]])


# NEURAL NETWORK
def get_NN(feature_num):
    """
    :param feature_num: number of input nodes
    :returns: The compiled neural network used for predicting
    """

    model = tf.keras.models.Sequential([

        tf.keras.layers.Dense(64, input_dim=feature_num, activation="relu"),

        tf.keras.layers.Dense(12, activation="relu"),

        # Set a dropout rate to avoid overfitting
        tf.keras.layers.Dropout(0.4),

        # Add an output layer with output units of dead or alive
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # Train neural network -
    model.compile(
        optimizer="adam",
        # It is a binary output
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# def try_NN():
#     # Get the num of features (should be 11)
#     model = get_NN(X_train.shape[1])
#     #print(X_train, y_train.values)
#     model.fit(X_train, y_train.values, validation_data=(
#         X_val, y_val), epochs=EPOCHS)
#     # model.fit(np.concatenate(X_train, X_val),
#     #   y_train.values + y_val.values, epochs=EPOCHS)

#     return model

# Logistic regression


def try_random_forests():
    # Random Forests
    random_forest = RandomForestClassifier(n_estimators=100, random_state=0)
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_val)
    print("Random Forests:", accuracy_score(y_val, y_pred))


def try_neighbours(neighbours):
    knn = KNeighborsClassifier(n_neighbors=neighbours)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)
    print("K nearest neighbours:", accuracy_score(y_val, y_pred))
    return accuracy_score(y_val, y_pred)


def try_gaussian_nb():
    gaussian = GaussianNB()
    gaussian.fit(X_train, y_train)
    y_pred = gaussian.predict(X_val)
    print("Gaussian NB:", accuracy_score(y_val, y_pred))


def try_svc():
    svc = SVC()
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_val)
    print("SVC: ", accuracy_score(y_val, y_pred))


def stacking_models():
    """
    Returns the stacked model
    """
    level0 = []  # First layer
    # Add the optimised hyperparameters.
    level0.append(('lr', LogisticRegression(
        solver='newton-cg', penalty='l2', C=0.1)))
    level0.append(('RF', RandomForestClassifier(
        n_estimators=50, min_samples_split=6, max_features='auto', max_depth=10)))
    level0.append(('GB', GaussianNB(var_smoothing=0.12328467394420659)))
    level0.append(('KN', KNeighborsClassifier(
        n_neighbors=8, p=1, weights='uniform')))
    level0.append(('gbc', GradientBoostingClassifier()))
    level0.append(('xgbc', XGBClassifier(verbosity=0)))

    # Specify the num of input parameters
    #level0.append(('NN', get_NN(11)))

    level1 = LogisticRegression(solver='liblinear')  # Second layer

    # Use 5-fold cross-validation when training it. The group that is not used on the base models are used to train the meta model, they are first passed through the base models, but the base models are not trained on them.
    model = StackingClassifier(
        estimators=level0, final_estimator=level1, cv=5, n_jobs=-1)

    return model


def evaluate_model(model, X, y):
    """
    Return
    -----------
    Return accracy score that perform base on Cross Validation
    """
    # split the data into 10 groups of equal size with 5 repeats of the process data is shuffled before each repetition so there are different groups

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)

    # Uses cross validation specified above to score the model passed in based on accuracy
    # n_jobs = -1 means use all processors in parallel when computing the scores and training the model.
    scores = cross_val_score(
        model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    print(np.mean(scores))
    return scores


# Find out the best parameters


def find_best_params(param, try_func):
    """
    :param param: List of paramaters to try
    :param try_func: passed to the machine learning function, which should retunr an accuracy score
    :returns: the param with the highest accuracy 
    """
    accuracy = 0
    best_param = 0
    for i in param:
        temp_acc = try_func(i)
        if temp_acc > accuracy:
            accuracy = temp_acc
            best_param = i

    return best_param


def get_best_logistic_params(cv, X, y):
    """
    Does hyperparameter optimization (parameters that control the learning process)
    cv used to get the params
    """
    lr = LogisticRegression(random_state=0)
    # Create a dict with
    space = {}
    # The solver used
    space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
    # the penalty used for regularisation (reduces the complexity by penalising weights) normally reduces overfitting
    space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
    # this is the inverse of regularisation, the higher the value the lower regularisation (so the higher overfitting)
    space['C'] = [0.001, 0.01, 0.1, 1, 10, 100]

    # Randomly search through the possible hyperparameters and get the bes ones based on accuracy - not the optimal as not gridsearchCV
    # Try 500 different value pairs
    search = GridSearchCV(
        lr, space, scoring='accuracy', n_jobs=-1, cv=cv)

    # execute search
    result = search.fit(X, y)

    # Score print
    print(f'Best Score of LogisticRegression: {result.best_score_}')
    print(f'Best Hyperparameters of LogisticRegression: {result.best_params_}')

    return result.best_params_


def get_best_tree_params(cv, X, y):
    rf = RandomForestClassifier()
    # define search space
    space = dict()
    # Number of trees used per decision
    space['n_estimators'] = [10, 50, 100, 300, 1000]
    # The maximum number of features the decision tree can take into account
    space['max_features'] = ['auto', 'sqrt', 'log2', None]
    # None specifies tree can go as deep as it wants
    #
    space['max_depth'] = [10, 20, 60, 100, None]

    space['min_samples_split'] = [2, 4, 6]

    # define search
    search = RandomizedSearchCV(
        rf, space, n_iter=20, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)
    # execute search
    result = search.fit(X, y)

    # Score print
    print('Best Score of RandomForestClassifier: %s' % result.best_score_)
    print('Best Hyperparameters of RandomForestClassifier: %s' %
          result.best_params_)


def get_best_bayes_parmas(cv, X, y):
    GB = GaussianNB()
    # define search space
    space = dict()

    # Generates 100 values uniformly spaced between 9 and -9 to try.

    space['var_smoothing'] = np.logspace(9, -9, num=100)
    # define search
    search = RandomizedSearchCV(
        GB, space, n_iter=20, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)
    # execute search
    result = search.fit(X, y)

    # Score print
    print('Best Score of GaussianNB: %s' % result.best_score_)
    print('Best Hyperparameters of GaussianNB: %s' % result.best_params_)


def get_best_k_neighbours(cv, X, y):
    kn = KNeighborsClassifier()
    # define search space
    space = dict()
    # Number of neighbours used when classifying
    space['n_neighbors'] = [3, 5, 8, 10]
    # Distance means algorithm takes into account how close the neighbours are for the weights.
    space['weights'] = ['uniform', 'distance']
    # 1 uses manhattan distance, 2 uses euclidean distance
    space['p'] = [1, 2]

    # define search
    search = RandomizedSearchCV(
        kn, space, n_iter=20, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)
    # execute search
    result = search.fit(X, y)

    # Score print
    print('Best Score of RandomForestClassifier: %s' % result.best_score_)
    print('Best Hyperparameters of RandomForestClassifier: %s' %
          result.best_params_)


X, X_test, y = prepare_data(False)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

# Get the best params - this was done before

#logistic_params = get_best_logistic_params(cv, X, y)
#get_best_tree_params(cv, X, y)
#get_best_bayes_parmas(cv, X, y)
#get_best_k_neighbours(cv, X, y)


#scores = evaluate_model(stacking_models(), X, y)

# print(scores)


predict_labels = predict_labels(
    stacking_models().fit(X, y), X_test, "submissions.csv")


# prepare_data()
