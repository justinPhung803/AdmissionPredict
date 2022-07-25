
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras import layers
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score




dataset=pd.read_csv("admissions_data.csv")
print(dataset.head())

features=dataset.iloc[:,1:8]
labels=dataset.iloc[:,-1]

features_train, features_test, labels_train, labels_test= train_test_split(features,labels, test_size=0.25, random_state=42)

features_train_scaled=StandardScaler().fit_transform(features_train)
features_test_scaled=StandardScaler().fit_transform(features_test)

def design_model(feature_data):
  my_model=Sequential()
  num_features=feature_data.shape[1]
  input=keras.Input(shape=(num_features))
  my_model.add(input)
  my_model.add(layers.Dense(16, activation='relu'))
  my_model.add(layers.Dropout(0.1))
  my_model.add(layers.Dense(8, activation='relu'))
  my_model.add(layers.Dropout(0.2))
  my_model.add(layers.Dense(1))

  opt = Adam(learning_rate=0.005)
  my_model.compile(optimizer=opt, loss='mse', metrics=['mae'])

  return my_model


my_model=design_model(features_train_scaled)


callback=EarlyStopping(monitor='val_loss',mode='min', verbose=1, patience=20)

history=my_model.fit(features_train_scaled, labels_train.to_numpy(), epochs= 100, batch_size=8, callbacks=[callback],validation_split=0.25,  verbose=1)

val_mse, val_mae= my_model.evaluate(features_test_scaled,labels_test.to_numpy(),verbose=0)

print("mse: ",val_mse)
print("mae: ",val_mae)

y_pred= my_model.predict(features_test_scaled)
print(r2_score(labels_test,y_pred))

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae'])
ax1.set_title('model mae')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')

# Plot loss and val_loss over each epoch
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')

plt.show()
