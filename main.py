# mlp for multiclass classification
import csv
from numpy import argmax
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras import regularizers
from sklearn.metrics import confusion_matrix
import numpy as np



path = '/home/mat/Desktop/cond_coop/data/sesca3.csv'
df = read_csv(path)

df = df.drop('choice1', 1)
df = df.drop('cooptyp.1', 1)
print(df.head(5))
print(df['cooptyp'].unique())
X = df.values[:, 1:]
y = df.values[:, 0]


X = X.astype('float32')
y = LabelEncoder().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
n_features = X_train.shape[1]
#print(y_train)
#print(y_test)
#print(X_train)
#print(X_test)
model = Sequential()
model.add(keras.layers.Flatten(input_shape=(n_features,)))
model.add(Dense(81, activation='relu', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.0001)))
model.add(Dropout(0.099))

model.add(Dense(6, activation='softmax'))

#model.summary()
plot_model(model, 'model.png', show_shapes=True)

# compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001), loss='sparse_categorical_crossentropy', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', patience=20)

# fit the model
history = model.fit(X_train, y_train, epochs=200, batch_size=74, verbose=0, validation_split=0.3, callbacks=[es])

# evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)

y_p = model.predict(X_test)
y_pred=[y_p[ar].tolist().index(max(y_p[ar])) for ar in range(len(y_p))]

#print(y_test)

con_mat=confusion_matrix(y_test, y_pred)
#print(confusion_matrix(y_test, y_pred).numpy()) #y_true, y_pred
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
con_mat_df = pd.DataFrame(con_mat_norm,
                     columns = ["cc","fr","ncc","uc","tc","other"])
print(con_mat_df)
#tnlgb, fplgb, fnlgb, tplgb = confusion_matrix(y_test, y_pred).ravel() #y_true, y_pred

#print("True negative: %s, False positive: %s, False negative: %s, True positive %s | share false %.2f" %(tnlgb, fplgb, fnlgb, tplgb, ((fplgb+fnlgb)/(fplgb+fnlgb+tplgb+tnlgb))))


print(history.history.keys())
pyplot.title('Learning Curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('Cross Entropy')
pyplot.plot(history.history['acc'], label='train')
pyplot.plot(history.history['val_acc'], label='val')
pyplot.legend()
pyplot.show()





