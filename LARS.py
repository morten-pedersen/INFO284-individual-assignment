import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing

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

	test_data.replace(["L", "M", "H"], [1, 2, 3], inplace = True)
	test_data = test_data.rename({'N level': 'n_level', 'Plant Weight(g)': 'weight'}, axis = 'columns')
	test = pd.get_dummies(test_data, columns = ['species'])
	X_test = test[test.columns.difference(['weight'])]
	y_test = test.iloc[:, 1]
	y_test = y_test.values
	print("Test data loaded from: \n {}".format(test_file),"\n")

	return X_test, y_test


open_training_file("Flaveria_train.csv")
open_test_file("Flaveria_test.csv")

scaler = preprocessing.MinMaxScaler()

x1_scaled = scaler.fit_transform(X_train)
x2_scaled = scaler.fit_transform(X_test)

Lars = linear_model.Lars()

Lars.fit(x1_scaled, y_train)

print("R2 score using training data: {:.4f}".format(Lars.score(x1_scaled, y_train)))
print("R2 score using test data: {:.4f}".format(Lars.score(x2_scaled, y_test)),"\n")

predictions = Lars.predict(x2_scaled)

print("Predicted weights: {}".format(predictions))
print("Actual weights: {}".format(y_test))

plt.scatter(predictions, y_test)
plt.xlabel("predictions")
plt.ylabel("y2")
plt.title("Score: {:.4f}".format(Lars.score(x2_scaled, y_test)))
plt.show()


