from src import data, analysis_of_dataset, bayesnet_probabilities, knn_cbn, kmeans
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

"""
    Reading the data from covid.csv
    Data should have following dimensions: 442822 rows x 33 columns
    Original data file: https://github.com/beoutbreakprepared/nCoV2019/tree/master/latest_data
"""
raw = data.loadData()
dataset = data.trim(raw)

analysis_of_dataset.plotRaw(data.trim(raw, norm=False))


# ================================================ #
# PART 1: ANALYSIS OF THE DATASET
# ================================================ #


# === Correlation === #
correlations = analysis_of_dataset.computeCorrelations(dataset)
print('Table of correlations')
print(correlations)
print('Correlations for outcome:')
print(correlations['outcome'])

# Getting the parameter that has highest correlation with 'outcome', excluding the result for 'outcome'x'outcome' (=1)
print('Best correlation with outcome: ' +
      str(correlations.index[correlations['outcome'] == correlations['outcome'].sort_values(ascending=False)[1]][0]) +
      ' (' + str(correlations['outcome'].sort_values(ascending=False)[1]) + ')')

# === PCA === #
analysis_of_dataset.pca(dataset)

# ================================================ #
# PART 2: BAYES NETS
# ================================================ #

# Loading new minimal data for this part

dataset = data.BN_data(raw)
bayesnet_probabilities.computeProbabilities(dataset)

# ================================================ #
# PART 3: MACHINE LEARNING
# ================================================ #

# KNN with confusion matrix: target is 'outcome'
dataset = data.trim(raw)
X = dataset.copy(deep=False).drop(columns=['outcome']).to_numpy()
Y = dataset['outcome'].to_numpy()


knn_cbn.plotKNNVariation(X, Y, max_k=1, iterations=6)


dataset = data.trim(raw)
X = dataset.copy(deep=False).drop(columns=['age']).to_numpy()
Y = dataset['age'].to_numpy()

X_train, X_test, Y_train, Y_test = knn_cbn.split_training_testing(X, Y)
print("LINEAR REGRESSION")
predictions = knn_cbn.show_regression(knn_cbn.fitLinRegression(X_train, Y_train, X_test, Y_test))
print("POLYNOMIAL REGRESSION")
predictions = knn_cbn.show_regression(knn_cbn.fitPolRegression(X_train, Y_train, X_test, Y_test, degree=2))

dataset = data.trim(raw).drop(['outcome'], axis=1)
kmeans.applyKmeans(dataset, 9)
# """
plt.show()
