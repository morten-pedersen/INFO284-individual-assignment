import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import neighbors
from pandas import DataFrame as df
from sklearn.preprocessing import LabelEncoder


file_handler = open("Flaveria.csv", "r")

dataset = pd.read_csv(file_handler, sep=',', header=0)
file_handler.close()

data = pd.DataFrame(dataset)

#dataset.replace(["L", "M", "H", "brownii", "pringlei", "trinervia", "ramosissima", "robusta", "bidentis"],
                   # [1, 2, 3, 1, 2, 3, 4, 5, 6], inplace=True)
#dataset = dataset.rename({'N level':'n_level', 'Plant Weight(g)': 'weight'}, axis='columns')

encoded = data.apply(LabelEncoder().fit_transform)
oh = preprocessing.OneHotEncoder()
data_enc = oh.fit_transform(encoded)

print(data_enc)


#onehotdataset = pd.get_dummies(dataset, columns=['n_level', 'species'])

#X = data_enc[data_enc.columns.difference(['weight'])]
#y = data_enc.iloc[:, 0]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#scaler = preprocessing.MinMaxScaler()

#x1_scaled = scaler.fit_transform(X_train)
#y1 = y_train.values.reshape(-1, 1)
#y1_scaled = scaler.fit_transform(y1)
#x2_scaled = scaler.fit_transform(X_test)
#y2 = y_test.values.reshape(-1, 1)
#y2_scaled = scaler.fit_transform(y2)