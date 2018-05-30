% matplotlib
inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import neighbors
from pandas import DataFrame as df
from sklearn.preprocessing import LabelEncoder
from IPython.display import display
from sklearn.decomposition import PCA

file_handler = open("Flaveria.csv", "r")

dataset = pd.read_csv(file_handler, sep = ',', header = 0)
file_handler.close()

data = pd.DataFrame(dataset)

dataset.replace(["L", "M", "H"],  #, "brownii", "pringlei", "trinervia", "ramosissima", "robusta", "bidentis"],
                [1, 2, 3], inplace = True)  #, 1, 2, 3, 4, 5, 6], inplace=True)
dataset = dataset.rename({'N level': 'n_level', 'Plant Weight(g)': 'weight'}, axis = 'columns')
#le = LabelEncoder()
#le.fit(data['N level', 'species'])
#transformed = le.transform(data['N level', 'species'])
#encoded = data.apply(le.fit_transform)
#enc = preprocessing.OneHotEncoder()
#data_enc = enc.fit_transform(encoded['species'])

#print(data_enc)

onehotdataset = pd.get_dummies(dataset, columns = ['species'])

#onehotdataset.head()

X = onehotdataset[onehotdataset.columns.difference(['weight'])]
y = onehotdataset.iloc[:, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 9)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#scaler = preprocessing.MinMaxScaler()

#x1_scaled = scaler.fit_transform(X_train)
#y1 = y_train.values.reshape(-1, 1)
#y1_scaled = scaler.fit_transform(y1)
#x2_scaled = scaler.fit_transform(X_test)
#y2 = y_test.values.reshape(-1, 1)
#y2_scaled = scaler.fit_transform(y2)

lasso = linear_model.Lars()

lasso.fit(X_train, y_train)


print(lasso.score(X_train, y_train))
score = lasso.score(X_test, y_test)
print(score)

predictions = ridge.predict(X_test)

plt.scatter(y_test, predictions)
plt.xlabel("True values")
plt.ylabel("Predictions")
plt.title("Score: {}".format(score))
plt.show()