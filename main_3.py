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
from sklearn.utils import class_weight
from tensorflow.keras.layers import LeakyReLU
from keras.layers import GaussianNoise
from keras.utils import to_categorical


path = '/home/mat/Desktop/cond_coop/data/sesca3_uc.csv'
df = read_csv(path)

df = df.drop('choice1', 1)
df = df.drop('cooptyp.1', 1)
print(df.head(2))
print(df['cooptyp'].unique())
X = df.values[:, 1:]
y = df.values[:, 0]
#y = to_categorical(y)

X = X.astype('float32')
y = LabelEncoder().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
n_features = X_train.shape[1]

#for one hot
#y_integers = np.argmax(y, axis=1)
#class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
#class_weights = dict(enumerate(class_weights))

class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)
class_weights = {i : class_weights[i] for i in range(3)}

######good models for sesca3
# model = Sequential()
# model.add(keras.layers.Flatten(input_shape=(n_features,)))
# model.add(Dense(171, activation='sigmoid', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.0001)))
# model.add(Dropout(0.4067645))
# model.add(Dense(333, activation='sigmoid', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.0001)))
# model.add(Dropout(0.0237))
# model.add(Dense(3, activation='softmax'))

######good models for sesca3_uc
# model = Sequential()
# model.add(keras.layers.Flatten(input_shape=(n_features,)))
# model.add(Dense(150, activation='sigmoid', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.001)))
# model.add(Dropout(0.38))
# model.add(Dense(130, activation='sigmoid', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.001)))
# model.add(Dropout(0.1))
# model.add(Dense(3, activation='softmax'))
##lr=0.001

#strong CC bias
# model = Sequential()
# model.add(keras.layers.Flatten(input_shape=(n_features,)))
# model.add(Dense(150, activation='relu', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.001)))
# model.add(Dropout(0.4))
# model.add(Dense(120, activation='relu', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.001)))
# model.add(Dropout(0.2))
# model.add(Dense(3, activation='softmax'))
##lr=0.001

model = Sequential()
model.add(keras.layers.Flatten(input_shape=(n_features,)))
model.add(GaussianNoise(0.03))
model.add(Dense(83, activation='relu', kernel_initializer='he_normal',kernel_regularizer=regularizers.l1_l2(l1=0.0001,l2=0.001)))
model.add(Dropout(0))
model.add(Dense(3, activation='softmax'))
#lr=0.049

#model.summary()
plot_model(model, 'model.png', show_shapes=True)

# compile the model
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', patience=50)

# fit the model
history = model.fit(X_train, y_train, epochs=200, batch_size=81, verbose=0, validation_split=0.3,  class_weight=class_weights)#,callbacks=[es])

# evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)

y_p = model.predict(X_test)
y_pred=[y_p[ar].tolist().index(max(y_p[ar])) for ar in range(len(y_p))]

#convert one hot to ordinal
# y_obs=[]
# for i in y_test:
#     i = i.tolist()
#     y_obs.append(i.index(1))

#con_mat=confusion_matrix(y_obs, y_pred)
con_mat=confusion_matrix(y_test, y_pred)
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
con_mat_df = pd.DataFrame(con_mat_norm,
                     columns = ["cc","fr","other"])
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





