import numpy as np
import librosa as lb
from librosa.feature import mfcc, delta
import glob
import os
import csv
import pandas as pd

input_dir = '../data/language/'
input_files = sorted(glob.glob(os.path.join(input_dir + '*.wav')))

def mfcc_mean_var(filename):
    
	data, rate = lb.load(filename)
	mfcc_data = mfcc(data,rate)[1:14]
	mfcc_delta = delta(mfcc_data,width=3)
	mfcc_delta2 = delta(mfcc_data,width=3,order=2)

	b = mfcc_data.mean(axis=1)
	b = np.append(b,mfcc_data.var(axis=1))
	b = np.append(b,mfcc_delta.mean(axis=1))
	b = np.append(b,mfcc_delta.var(axis=1))
	b = np.append(b,mfcc_delta2.mean(axis=1))
	b = np.append(b,mfcc_delta2.var(axis=1))
	b = np.append(b,'Language_label')
	b = b.reshape(1,79)
	return b

a = np.zeros((1,79), dtype = float)
a.reshape(1,79)
for index,file in enumerate(input_files):
	x = mfcc_mean_var(file)
	a = np.append(a,x,axis=0)
	print(index)

a = a[1:,:]
pd.DataFrame(a).to_csv('../Output/language_label.csv',header=None, index=None)
print(a.shape)

# from google.colab import drive
# drive.mount('/content/drive')
# %cd 'drive/My Drive/Colab Notebooks/Lang_class'

