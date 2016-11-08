# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd
import numpy as np

# DictVectorizer
from sklearn.feature_extraction import DictVectorizer as DV

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Get a matrix of numerical data
num_cols = ['T1_V1', 'T1_V2', 'T1_V3', 'T1_V10', 'T1_V13', 'T1_V14', 'T2_V1', 'T2_V2', 'T2_V4', 'T2_V6', 'T2_V7', 'T2_V8', 'T2_V9', 'T2_V10', 'T2_V14', 'T2_V15']
x_num_train = train[num_cols].as_matrix()

# Scale numerical data to <0,1>
max_train = np.amax(x_num_train, 0) # row vector of max values per column
x_num_train = x_num_train / max_train

# Convert Panda data frame to a list of dicts
cat_train = train.drop(num_cols + ['Id', 'Hazard'], axis = 1)
x_cat_train = cat_train.T.to_dict().values()

# Vectorize:
# Convert categorical data to numerical data
vectorizer = DV( sparse = False )
vec_x_cat_train = vectorizer.fit_transform( x_cat_train )

# Finalize x_train and y_train
x_train = np.hstack((x_num_train, vec_x_cat_train)) # concatenate matrices horizontally
y_train = train.Hazard





















