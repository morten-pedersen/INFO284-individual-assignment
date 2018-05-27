import numpy as np
import pandas as pd
import scipy.stats as stats
from pandas import DataFrame as df
from sklearn import preprocessing
import matplotlib
from sklearn import svm
from sklearn.model_selection import train_test_split


file_handler = open("Flaveria.csv", "r")

# parses the csv data into a pandas data frame
dataset = pd.read_csv(file_handler, sep=',', header=0)

dataset.replace(["L", "M", "H", "brownii", "pringlei", "trinervia", "ramosissima", "robusta", "bidentis"],
                    [1, 2, 3, 1, 2, 3, 4, 5, 6], inplace=True)
dataset = dataset.rename({'N level':'n_level', 'Plant Weight(g)': 'weight'}, axis='columns')

file_handler.close()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 9) # 69 gives 0.272

clf = svm.SVR()
clf.fit(X_train, y_train)