import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy.stats as stats
from pandas import DataFrame as df
from sklearn import preprocessing
import matplotlib
matplotlib.style.use('ggplot')

file_handler = open("Flaveria.csv", "r")

# parses the csv data into a pandas data frame
dataset = pd.read_csv(file_handler, header=0)

dataset.replace(["L", "M", "H", "brownii", "pringlei", "trinervia", "ramosissima", "robusta", "bidentis"],
                    [1, 2, 3, 1, 2, 3, 4, 5, 6], inplace=True)
dataset = dataset.rename({'N level':'n_level', 'Plant Weight(g)': 'weight'}, axis='columns')

file_handler.close()

scaler = preprocessing.MinMaxScaler()
scaleddata = scaler.fit_transform(dataset)
scaleddata = pd.DataFrame(scaleddata, columns=['n_level', 'species', 'weight'])

from sklearn.model_selection import train_test_split
X = scaleddata.iloc[:, :-1].values
y = scaleddata.iloc[:, 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 9)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_predicted = regressor.predict(X_test)

print(regressor.score(X_test, y_test))