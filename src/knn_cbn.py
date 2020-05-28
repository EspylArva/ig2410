import numpy
from sklearn import metrics, neighbors, preprocessing, linear_model, model_selection
from numpy import unique, asarray
from random import randint
import matplotlib.pyplot as plt
import numpy as np




#    Splitting data and Linear Regression
#    https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f


"""
    input: dataset, target, splitting ratio, seed
    output: training_data, testing_data, training_target, expected_prediction
    --
    We split the data following a X/Y ratio:
    X is the ratio of data used for training
    Y is the ratio of data used for testing
    The seed is used to randomize the splitting of data. We can set results by setting the seed.
"""


def split_training_testing(X, Y, test_size=0.2, seed=randint(0, 100)):
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
    lab_enc = preprocessing.LabelEncoder()
    Y_train = lab_enc.fit_transform(Y_train)
    Y_test = lab_enc.fit_transform(Y_test)
    return X_train, X_test, Y_train, Y_test


"""
    input: training_set, training_target, number_of_neighbors
    output: knn_model
    --
    To use the KNN algorithm, we have to use the settings.
"""


def fitTNN(X_train, Y_train, n_neighbors=10):
    knn = neighbors.KNeighborsClassifier(n_neighbors)
    knn.fit(X_train, Y_train)
    return knn


"""
    input: testing_set, expected_prediction, knn_model
    output: results of the prediction
    --
    Using the KNN settings, we can predict the target. The function outputs one array per expected result (from the testing set).
    The array contains the prediction, the distance between the prediction and the actual goal, and the state of the prediction.
"""


def predictTNN(X_test, Y_test, knn):
    res = []
    for int_i in range(len(X_test)):
        entry_ = X_test[int_i]
        target_ = Y_test[int_i]
        prediction = knn.predict([entry_])
        dist = metrics.accuracy_score([target_], prediction)

        #  I decided that a positive is a death, whereas a negative is a patient getting discharged.
        result_confusion = 'true ' if target_ == prediction else 'false '
        result_confusion += 'positive' if prediction == 1 else 'negative'

        res.append([prediction[0], dist, result_confusion])
    return res


"""
    input: training_data, testing_data, training_target, expected_prediction, number_of_neighbors
    output: results of the prediction
    --
    We train the model, then predict outputs for the testing set using the training set.
"""


def TNN(X_train, X_test, Y_train, Y_test, n_neighbors):
    knn = fitTNN(X_train, Y_train, n_neighbors)
    predictions = predictTNN(X_test, Y_test, knn)
    return asarray(predictions)


"""
    input: array containing distances from prediction to expectation
    output: average accuracy of the KNN predictions
    --
    We return the mean of the accuracy of every prediction
"""


def percentageTNNGoodGuesses(distance_pred_to_goal):
    return distance_pred_to_goal.astype(numpy.float).sum() / len(distance_pred_to_goal)


"""
    input: state of the predictions
    output: count for each state
    --
    For every prediction, there is an associated state (https://en.wikipedia.org/wiki/Confusion_matrix). This function
    returns a count of every state.
"""


def TNNConfusionMatrix(confusion):
    uni, counts = unique(confusion, return_counts=True)
    return dict(zip(uni, counts))


"""
    input:
    output:
    --
    
"""


def show_regression(results):
    [Y_testing, predictions] = results
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(8)

    x = np.arange(len(Y_testing))
    width = 0.35

    ax.bar(x - width / 2, Y_testing, width, label='Targets')
    ax.bar(x + width / 2, predictions, width, label='Predictions')
    ax.set_xlabel("Age Predictions")
    ax.set_ylabel("Age")
    ax.legend(loc=1)
    return predictions


def fitLinRegression(X_training, Y_training, X_testing, Y_testing):
    model = linear_model.LinearRegression()
    model.fit(X_training, Y_training)
    predictions = model.predict(X_testing)
    mse = metrics.mean_squared_error(Y_testing, predictions)

    print({"Slope:": model.coef_, "Intercept": model.intercept_})
    print(metrics.r2_score(Y_testing, predictions))
    print('Rooted Mean Squared Error:', np.sqrt(mse))
    print('Mean Squared Error:', mse)

    return [Y_testing, predictions]


def fitPolRegression(X_training, Y_training, X_testing, Y_testing, degree=2):
    polynomial_features = preprocessing.PolynomialFeatures(degree=degree)
    x_poly_train = polynomial_features.fit_transform(X_training)
    x_poly_test = polynomial_features.fit_transform(X_testing)
    return fitLinRegression(x_poly_train, Y_training, x_poly_test, Y_testing)

# WORKING ON: MSE
"""
    https://medium.com/datadriveninvestor/regression-in-machine-learning-296caae933ec
    https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f
    https://towardsdatascience.com/statistics-for-machine-learning-r-squared-explained-425ddfebf667
    https://realpython.com/numpy-scipy-pandas-correlation-python/
"""

"""
    input: dataset, targets, number_of_neighbors, seed
    output: plot of the variation of accuracy of KNN predictions, depending on the number of neighbors
    --
    In order to find the best number K of neighbors to use in the KNN algorithm, we can vary the number of neighbors and
    plot the accuracy of each KNN prediction. Modifying the seed splits the dataset differently and is useful to further
    remove biases from the search of the best K.
"""


def rangeKNN(X, Y, MAX_K_NEIGHBOUR=10, seed=randint(0, 100), index=0, axes=None):
    X_train, X_test, Y_train, Y_test = split_training_testing(X, Y, seed=seed)
    accuracies = []
    for k in range(1, MAX_K_NEIGHBOUR + 1):
        knn_res = TNN(X_train, X_test, Y_train, Y_test, k)
        accuracy = percentageTNNGoodGuesses(knn_res[:, 1])
        print("value of k (number of neighbours): " + str(k) + " - accuracy of KNN: " + str(accuracy))
        print(TNNConfusionMatrix(knn_res[:, 2]))
        accuracies.append(accuracy)

    row = int(index / 3)
    col = index % 3

    axes[row, col].plot(range(1, MAX_K_NEIGHBOUR + 1), accuracies, '.r-')
    axes[row, col].set_xlabel("Number of Nearest Neighbours")
    axes[row, col].set_ylabel("Accuracy")
    axes[row, col].set_xticks([])
    axes[row, col].set_title('K-NN (seed: ' + str(seed) + ')')


def plotKNNVariation(X, Y, max_k=10, iterations=9):
    fig, axes = plt.subplots(numpy.math.ceil(iterations / 3), min(iterations, 3))
    fig.set_figheight(3 * numpy.math.ceil(iterations / 3))
    fig.set_figwidth(3 * min(iterations, 3))

    for i in range(iterations):
        rndValue = randint(0, 100)
        rangeKNN(X, Y, max_k, seed=rndValue, index=i, axes=axes)
    plt.draw()
