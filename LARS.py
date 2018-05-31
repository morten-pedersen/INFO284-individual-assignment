import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import neighbors
from pandas import DataFrame as df
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA




def open_training_file(training_file):

	global X_train
	global y_train

	file_handler = open(training_file, "r")

	opened_train_data = pd.read_csv(file_handler, sep = ',', header = 0)
	file_handler.close()

	train_data = pd.DataFrame(opened_train_data)

	train_data.replace(["L", "M", "H"],  #, "brownii", "pringlei", "trinervia", "ramosissima", "robusta", "bidentis"],
	                   [1, 2, 3], inplace = True)  #, 1, 2, 3, 4, 5, 6], inplace=True)
	train_data = train_data.rename({'N level': 'n_level', 'Plant Weight(g)': 'weight'}, axis = 'columns')
	train = pd.get_dummies(train_data, columns = ['species'])
	X_train = train[train.columns.difference(['weight'])]
	y_train = train.iloc[:, 1]
	return X_train, y_train

	print("Training data loaded from: \n {}".format(training_file))


def open_test_file(test_file):

	global X_test
	global y_test

	file_handler = open(test_file, "r")

	opened_test_data = pd.read_csv(file_handler, sep = ',', header = 0)
	file_handler.close()

	test_data = pd.DataFrame(opened_test_data)

	test_data.replace(["L", "M", "H"],  #, "brownii", "pringlei", "trinervia", "ramosissima", "robusta", "bidentis"],
	                  [1, 2, 3], inplace = True)  #, 1, 2, 3, 4, 5, 6], inplace=True)
	test_data = test_data.rename({'N level': 'n_level', 'Plant Weight(g)': 'weight'}, axis = 'columns')
	test = pd.get_dummies(test_data, columns = ['species'])
	X_test = test[test.columns.difference(['weight'])]
	y_test = test.iloc[:, 1]
	return X_test, y_test
	print("Test data loaded from: \n {}".format(test_file))


open_training_file("Flaveria_train.csv")
open_test_file("Flaveria_test.csv")

Lars = linear_model.Lars()

Lars.fit(X_train, y_train)

print(Lars.score(X_train, y_train))
score = Lars.score(X_test, y_test)
print(score)

predictions = Lars.predict(X_test)

plt.scatter(y_test, predictions)
plt.xlabel("True values")
plt.ylabel("Predictions")
plt.title("Score: {}".format(score))
plt.show()

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0., random_state=9)


#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)

#scaler = preprocessing.MinMaxScaler()

#x1_scaled = scaler.fit_transform(X_train)
#y1 = y_train.values.reshape(-1, 1)
#y1_scaled = scaler.fit_transform(y_train)
#x2_scaled = scaler.fit_transform(X_test)
#y2 = y_test.values.reshape(-1, 1)
#y2_scaled = scaler.fit_transform(y_test)