#%matplotlib inline
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy.stats as stats
from pandas import DataFrame as df



file_handler = open("Flaveria.csv", "r")

# parses the csv data into a pandas data frame
data = pd.read_csv(file_handler, sep = ",")
data_column = data.rename({'N level':'n_level', 'Plant Weight(g)': 'weight'}, axis='columns')
data_replaced_flowernames = data_column.replace({'species':{'brownii': 1, 'pringlei': 2, 'trinervia': 3, 'ramosissima': 4,
	'robusta': 5, 'bidentis': 6}})
dataset = data_replaced_flowernames.replace({'n_level':{'L': 1, 'M': 5, 'H': 10}})

file_handler.close()