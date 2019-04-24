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

# fix random seed for reproducibility

seed = 7
numpy.random.seed(seed)

# Data Loading

df = pd.read_csv('../ip_data/training_data.csv')
df = df.values
X = df[:,0:78].astype(float)
Y = df[:,78]
# Y.shape

# Normalization
X = (X - X.min(0)) / X.ptp(0)
# X.shape

# data encoding
# encode class values as integers

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
encoded_Y.shape

d = {i:j for i,j in zip(Y,encoded_Y)}
print(d)
# convert integers to dummy variables (i.e. one hot encoded)
y_vec = np_utils.to_categorical(encoded_Y)

x_train, x_test, y_train, y_test = train_test_split(X, y_vec, test_size=0.25, random_state=42)


# define baseline model

def baseline_model():
    # create model

    model = Sequential()
    model.add(Dense(500, input_dim=78))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.5))
    model.add(Dense(250))
    model.add(LeakyReLU(alpha=0.015))
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.5))
    model.add(Dense(12, activation='softmax'))
    
    
    # Compile model
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Model Training

model = baseline_model()
print(model.summary())
history = model.fit(x = x_train, y = y_train, validation_data = (x_test,y_test), epochs=200, batch_size=36, verbose=1, shuffle=1)

# from keras.models import load_model

# # Saving model
# # %cd '/content/drive/My Drive/Colab Notebooks/Lang_class/Model'
# # model.save('test5.h5')

# # Loading model
# load_model('test.h5')

import matplotlib.pyplot as plt

# Accuracy History

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('model_acc.png')

# Loss History

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

scores = model.evaluate(x_test,y_test, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

prediction = model.predict(x_test)
y_pred = (prediction>0.5)
matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
# matrix

# from google.colab import drive
# drive.mount('/content/drive')
# %cd 'drive/My Drive/Colab Notebooks/Lang_class/Code'

