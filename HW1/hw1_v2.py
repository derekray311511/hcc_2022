from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import seaborn as sns
from NeuralNetwork import *

# Import dataset
dataset = pd.read_csv('energy_efficiency_data.csv')
## There are 8 features and 768 data
## Use 75% of data to train, 25% to test
X = dataset.iloc[:, 0:-2]
Y = dataset.iloc[:, 8]
# print(X)
# print(Y)

## Convert the column into categorical columns
## 把名字(類別)轉換成數字表示
feature_5 = pd.get_dummies(X['Orientation'], drop_first=True)
feature_7 = pd.get_dummies(X['Glazing Area Distribution'], drop_first=True)
## Drop the state coulmn
X = X.drop('Orientation', axis = 1)
X = X.drop('Glazing Area Distribution', axis = 1)
## concat the dummy variables
X = pd.concat([feature_5, X], axis = 1)
X = pd.concat([feature_7, X], axis = 1)
X_plot = X
Y_plot = Y
# print(X)
## Transform dataframe to data
X = X.values
Y = Y.values
X_plot = X_plot.values
Y_plot = Y_plot.values
# print(X)
## Shuffle the data
X, Y = shuffle(X, Y)
# print(X)

def feature_normalize(X):
    # mean of indivdual column, hence axis = 0
    mu = np.mean(X, axis = 0)
    # Notice the parameter ddof (Delta Degrees of Freedom)  value is 1
    # Standard deviation (can also use range)
    sigma = np.std(X, axis = 0, ddof = 1)
    X_norm = (X - mu)/sigma
    return X_norm, mu, sigma

X, mu, sigma = feature_normalize(X)
# y, mu, sigma = feature_normalize(y)

## Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size = 0.25, random_state = 0)

# network
NN_regression = Network()
NN_regression.add(Dense(14, 140))
NN_regression.add(leakyReLu())
NN_regression.add(Dense(140, 1))

NN_regression.use(mse, mse_prime)

# train

error_hist, test_error_hist = NN_regression.fit(
    X_train, Y_train, X_test, Y_test, epochs=700, batch_size=5, learning_rate=0.0033)

# Plot Mean Square Error History
plt.figure()
plt.plot(error_hist)
plt.plot(test_error_hist)
plt.legend(['Train Error', 'Valid Error'])
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.savefig("Results/AllFea/ErrorHist")
plt.tight_layout()
# plt.show()

# Performance
prediction = np.array(NN_regression.predict(X_test)).reshape((1, -1))
print('RMSE =', rmse(Y_test, prediction))

# Save the Result
X_plot_train = X_plot[:576, :]
X_plot_test = X_plot[577:, :]
Y_plot_train = Y_plot[:576]
Y_plot_test = Y_plot[577:]

X_plot_train, mu, sigma = feature_normalize(X_plot_train)
X_plot_test, mu, sigma = feature_normalize(X_plot_test)

## train reslut
trainPrediction = np.array(NN_regression.predict(X_plot_train)).reshape((-1, 1))
dataframe_trainResult = pd.DataFrame(trainPrediction, columns=['Result'])
dataframe_Y_train = pd.DataFrame(Y_plot_train, columns=['Label'])
dataframe_trainResult = pd.concat([dataframe_trainResult, dataframe_Y_train], axis = 1)
dataframe_trainResult.to_csv("Results/AllFea/trainResult.csv")

## test result
testPrediction = np.array(NN_regression.predict(X_plot_test)).reshape((-1, 1))
dataframe_testResult = pd.DataFrame(testPrediction, columns=['Result'])
dataframe_Y_test = pd.DataFrame(Y_plot_test, columns=['Label'])
dataframe_testResult = pd.concat([dataframe_testResult, dataframe_Y_test], axis = 1)
dataframe_testResult.to_csv("Results/AllFea/testResult.csv")

## Plot result of train and test
plt.figure()
plt.plot(trainPrediction)
plt.plot(Y_plot_train)
# plt.subplot(211)
plt.legend(['Predict', 'Label'])
plt.xlabel("#th case")
plt.ylabel("Heating Load")
plt.title('Prediction for training data')
plt.savefig("Results/AllFea/PredForTrain")

plt.figure()
plt.plot(testPrediction)
plt.plot(Y_plot_test)
# plt.subplot(212)
plt.legend(['Predict', 'Label'])
plt.xlabel("#th case")
plt.ylabel("Heating Load")
plt.title('Prediction for testing data')
plt.savefig("Results/AllFea/PredForTest")

plt.tight_layout()
plt.show()

# ===================================== PART C ===================================== #
# Correlation Matrix with Heatmap

# X = dataset.iloc[:, :8]  #independent columns
# y = dataset.iloc[:, -1]  #target column i.e price range
# get correlations of each features in dataset
corrmat = dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(14, 8))
# plot heat map
ax = sns.heatmap(
    corrmat, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    annot=True,
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
plt.savefig("Results/AllFea/CorrHeatMap")
# plt.show()