# import tensorflow as tf
import numpy as np
import pandas as pd
# from sklearn.preprocessing import LabelEncoder

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere."""
    e = np.zeros((10))
    e[j] = 1.0
    return e

def read_dataset(input_file = 'kaggle_data/train.csv'):
	rawdata = pd.read_csv(input_file)
	X = rawdata[rawdata.columns[1:]].values
	y = rawdata[rawdata.columns[0]].values
	Y = np.array([vectorized_result(num-1) for num in y])
	# print (X.shape)
	# print (Y.shape)
	return X,Y


X,Y = read_dataset()


