#Mounting Drive

# from google.colab import drive
# drive.mount('/content/drive')
# %cd 'drive/My Drive/Colab Notebooks/Lang_class/Code'

# Importing Libraries

import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# Data Loading

df = pd.read_csv('../ip_data/testing_data.csv',header=None)
df1 = pd.read_csv('../ip_data/training_data.csv',header=None)


df1 = df1.values
X_train = df1[:,0:78].astype(float)
Y_train = df1[:,78]

df = df.values
X_test = df[:,0:78].astype(float)
Y_test = df[:,78]
# print(df1.shape,df.shape)

# Normalization
X_test = (X_test - X_train.min(0)) / X_train.ptp(0)
# x.shape


# data encoding

# encode class values as integers

encoder = LabelEncoder()
encoder.fit(Y_test)
encoded_Y = encoder.transform(Y_test)

d = {i:j for i,j in zip(Y_test,encoded_Y)}
print(d)

# convert integers to dummy variables (i.e. one hot encoded)
Y_vec = np_utils.to_categorical(encoded_Y)

from keras.models import load_model

# Loading model
model = load_model('../Model/test5.h5')


prediction = model.predict(Y_test)
Y_pred = (prediction>0.4)
matrix = confusion_matrix(Y_vec.argmax(axis=1), Y_pred.argmax(axis=1))
scores = model.evaluate(X_test,Y_vec, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
matrix

# Architecture plot

# from keras.utils import plot_model
# import matplotlib.pyplot as plt
# plot_model(model, to_file='model.png')