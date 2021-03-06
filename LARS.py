import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing


# function that opens the desired training file with the filename
# as argument, in this case "Flaveria_train.csv".
# replaces N level characters with integers:
# we can use 1, 2, 3 as representations for the N level if we assume that the average of high and low is medium,
# just as the average of 1 and 3 = 2

# the train and test files have been split manually, with 6 data points serving as test data, because sklearn's
# train_test_split function could sometimes remove all the L/M/H samples of one single species which negatively impacted
# the r2 score

def open_training_file(training_file):
	global X_train
	global y_train

	file_handler = open(training_file, "r")
	opened_train_data = pd.read_csv(file_handler, sep = ',', header = 0)
	file_handler.close()
	train_data = pd.DataFrame(opened_train_data)

	# replaces categorical values in "N level" with numerical data
	train_data.replace(["L", "M", "H"], [1, 2, 3], inplace = True)
	train_data = train_data.rename({'N level': 'n_level', 'Plant Weight(g)': 'weight'}, axis = 'columns')

	# pd.get_dummies uses one hot encoding to transform strings to binary columns, one for each species
	train = pd.get_dummies(train_data, columns = ['species'])

	# selects "n_level" and "species" for X, and leaves index 1 ("weight") for y
	X_train = train[train.columns.difference(['weight'])]
	y_train = train.iloc[:, 1]
	y_train = y_train.values
	print("Training data loaded from: \n {}".format(training_file))

	return X_train, y_train


def open_test_file(test_file):
	global X_test
	global y_test

	file_handler = open(test_file, "r")

	opened_test_data = pd.read_csv(file_handler, sep = ',', header = 0)
	file_handler.close()

	test_data = pd.DataFrame(opened_test_data)

	# replaces categorical values in "N level" with numerical data
	test_data.replace(["L", "M", "H"], [1, 2, 3], inplace = True)
	test_data = test_data.rename({'N level': 'n_level', 'Plant Weight(g)': 'weight'}, axis = 'columns')

	# pd.get_dummies uses one hot encoding to transform strings to binary columns, one for each species
	test = pd.get_dummies(test_data, columns = ['species'])

	# selects "n_level" and "species" for X, and leaves index 1 ("weight") for y
	X_test = test[test.columns.difference(['weight'])]
	y_test = test.iloc[:, 1]
	y_test = y_test.values
	print("Test data loaded from: \n {}".format(test_file), "\n")

	return X_test, y_test


# choose which CSV files to load training and test data from
open_training_file("Flaveria_train.csv")
open_test_file("Flaveria_test.csv")

# scales X using MinMax
scaler = preprocessing.MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# data is already normalized, so use normalize = False
Lars = linear_model.Lars(normalize = False)

Lars.fit(X_train_scaled, y_train)

print("R2 score on training data using LARS: {:.4f}".format(Lars.score(X_train_scaled, y_train)))
print("R2 score on test data using LARS: {:.4f}".format(Lars.score(X_test_scaled, y_test)), "\n")

predictions = Lars.predict(X_test_scaled)

print("Predicted weights: {}".format(predictions))
print("Actual weights: {}".format(y_test))

plt.scatter(predictions, y_test)
plt.xlabel("x")
plt.ylabel("y")
plt.title("LARS score: {:.4f}".format(Lars.score(X_test_scaled, y_test)))
plt.show()
