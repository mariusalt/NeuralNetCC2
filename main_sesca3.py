# mlp for multiclass classification
from numpy import argmax
import pandas as pd
import csv
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
import kerastuner as kt
import tensorflow as tf
from sklearn import preprocessing
import skopt
from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt import gp_minimize, forest_minimize
import tensorboard as tb
import tensorboard.program
import tensorboard.default
from keras.callbacks import TensorBoard
from keras import backend as K
from tensorflow.keras import regularizers
from datetime import datetime
import numpy as np
from sklearn.utils import class_weight
from keras.layers import GaussianNoise

log_dir_name = 'home/mat/Desktop/trials/'

start_date = datetime.now()
start_date = start_date.strftime("%d/%m/%Y %H:%M:%S")

# load the dataset
path = '/home/mat/Desktop/cond_coop/data/sesca3.csv'
df = read_csv(path)
# print(df1.head(3))
# split into input and output columns
# df = df1.drop(df1.columns[0:19], axis=1)
print(df.head(5))
print(df['cooptyp'].unique())
df = df.drop('choice1', 1)
df = df.drop('cooptyp.1', 1)
print(list(df))
X = df.values[:, 1:]
y = df.values[:, 0]

# ensure all data are floating point values
X = X.astype('float32')
# encode strings to integer
y = LabelEncoder().fit_transform(y)
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# determine the number of input features
n_features = X_train.shape[1]
# define weights
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
class_weights = {i : class_weights[i] for i in range(3)}


out=[]
##########l1_l2


for i in range(20):
    for lay in range(1, 5):
        dim_learning_rate = Real(low=1e-6, high=1e-1, prior='uniform', name='learning_rate')
        dim_activation = Categorical(categories=['relu', 'sigmoid','tanh','LeakyReLU'], name='activation')
        #  dim_regularization = Real(low=0.00001, high=0.01, prior='uniform', name='regularization')
        dim_regularization1 = Categorical(categories=[0.01, 0.001, 0.0001, 0.00001], name='regularization1')
        dim_regularization2 = Categorical(categories=[0.01, 0.001, 0.0001, 0.00001], name='regularization2')
        dim_noise = Categorical(categories=[0.00, 0.025, 0.05, 0.1,0.2], name='noise')
        dim_num_dense_nodes1 = Integer(low=5, high=512, name='num_dense_nodes1')
        dim_num_dense_nodes2 = Integer(low=5, high=512, name='num_dense_nodes2')
        dim_num_dense_nodes3 = Integer(low=5, high=512, name='num_dense_nodes3')
        dim_num_dense_nodes4 = Integer(low=5, high=512, name='num_dense_nodes4')
        dim_num_dense_nodes5 = Integer(low=5, high=512, name='num_dense_nodes5')
        dim_num_dropouts1 = Real(low=0, high=0.5, prior='uniform', name='num_dropouts1')
        dim_num_dropouts2 = Real(low=0, high=0.5, prior='uniform', name='num_dropouts2')
        dim_num_dropouts3 = Real(low=0, high=0.5, prior='uniform', name='num_dropouts3')
        dim_num_dropouts4 = Real(low=0, high=0.5, prior='uniform', name='num_dropouts4')
        dim_num_dropouts5 = Real(low=0, high=0.5, prior='uniform', name='num_dropouts5')
        dim_num_batch_size = Integer(low=5, high=100, name='num_batch_size')

        dimensions = [dim_learning_rate,
                      dim_activation,
                      dim_regularization1,
                      dim_regularization2,
                      dim_noise,
                      dim_num_batch_size,
                      dim_num_dense_nodes1,
                      dim_num_dropouts1,
                      dim_num_dense_nodes2,
                      dim_num_dropouts2,
                      dim_num_dense_nodes3,
                      dim_num_dropouts3,
                      dim_num_dense_nodes4,
                      dim_num_dropouts4,
                      dim_num_dense_nodes5,
                      dim_num_dropouts5,
                      ]
        dimensions = dimensions[:(lay + lay + 6)]
        print(dimensions)


        default_parameters5 = [1e-5, 'relu', 0.001,0.001,0.00, 32, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0]
        default_parameters4 = [1e-5, 'relu', 0.001,0.001,0.00, 32, 70, 0, 70, 0, 70, 0, 70, 0]
        default_parameters3 = [1e-5, 'relu', 0.001,0.001,0.00, 32, 70, 0, 70, 0, 70, 0]
        default_parameters2 = [1e-5, 'relu', 0.001,0.001,0.00, 32, 70, 0, 70, 0]
        default_parameters1 = [1e-5, 'relu', 0.001,0.001,0.00, 32, 70, 0]

        def_para = ['none', default_parameters1, default_parameters2, default_parameters3, default_parameters4, default_parameters5]

        if lay == 4:
            def create_model(learning_rate, activation, regularization1,regularization2,noise, num_dense_nodes1, num_dense_nodes2,
                             num_dense_nodes3, num_dense_nodes4,
                             num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4
                             ):
                model = Sequential()

                model.add(InputLayer(input_shape=(n_features,)))

                for i in range(lay):
                    name = 'layer_dense_{0}'.format(i + 1)
                    l1 = [num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4]
                    l2 = [num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4]
                    # add dense layer
                    model.add(GaussianNoise(noise))
                    model.add(Dense(l1[i],
                                    activation=activation,
                                    name=name, kernel_regularizer=regularizers.l1_l2(l1=regularization1,l2=regularization2)))
                    model.add(tf.keras.layers.Dropout(l2[i]))

                # use softmax-activation for classification.
                model.add(Dense(3, activation='softmax'))

                # Use the Adam method for training the network.
                optimizer = keras.optimizers.SGD(lr=learning_rate)

                # compile the model so it can be trained.
                model.compile(optimizer=optimizer,
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])

                return model


            @use_named_args(dimensions=dimensions)
            def fitness(learning_rate, activation,  regularization1,regularization2,noise, num_dense_nodes1, num_dense_nodes2, num_dense_nodes3,
                        num_dense_nodes4,
                        num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4
                        , num_batch_size):
                model = create_model(learning_rate=learning_rate,
                                     activation=activation,
                                     regularization1=regularization1,
                                     regularization2=regularization2,
                                     noise=noise,
                                     num_dense_nodes1=num_dense_nodes1,
                                     num_dense_nodes2=num_dense_nodes2,
                                     num_dense_nodes3=num_dense_nodes3,
                                     num_dense_nodes4=num_dense_nodes4,
                                     num_dropouts1=num_dropouts1, num_dropouts2=num_dropouts2,
                                     num_dropouts3=num_dropouts3,
                                     num_dropouts4=num_dropouts4)
                callback_log = TensorBoard(
                    #      log_dir=log_dir,
                    histogram_freq=0,
                    write_graph=True,
                    write_grads=False,
                    write_images=False)
                es = EarlyStopping(monitor='val_accuracy', patience=10)
                # Use Keras to train the model.
                history = model.fit(x=X_train,
                                    y=y_train,
                                    epochs=200,
                                    batch_size=num_batch_size,
                                    validation_data=(X_test, y_test),
                                    callbacks=[callback_log], class_weight=class_weights)
                accuracy = history.history['val_accuracy'][-1]
                print()
                print("Accuracy: {0:.2%}".format(accuracy))
                print()
                global best_accuracy
                best_accuracy=0.4
                if accuracy > best_accuracy:
                    # Save the new model to harddisk.
                    model.save('home/mat/Desktop/trials')
                    out.append([learning_rate, num_dense_nodes1, lay, activation, regularization1,regularization2,noise,
                                num_dense_nodes2,
                                num_dense_nodes3,
                                num_dense_nodes4,
                                num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                                'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none',
                                num_batch_size, accuracy])
                    best_accuracy = accuracy
                del model
                K.clear_session()
                return -accuracy


            default_parameters = def_para[lay]
            print(default_parameters)
            fitness(x=default_parameters)

            try:
                search_result = gp_minimize(func=fitness,
                                            dimensions=dimensions,
                                            acq_func='EI',  # Expected Improvement.
                                            n_calls=12,
                                            random_state=1234)
            except ValueError:
                pass
        elif lay == 3:
            def create_model(learning_rate, activation, regularization1,regularization2,noise, num_dense_nodes1, num_dense_nodes2,
                             num_dense_nodes3,
                             num_dropouts1, num_dropouts2, num_dropouts3):
                model = Sequential()

                model.add(InputLayer(input_shape=(n_features,)))

                for i in range(lay):
                    name = 'layer_dense_{0}'.format(i + 1)
                    l1 = [num_dense_nodes1, num_dense_nodes2, num_dense_nodes3]
                    l2 = [num_dropouts1, num_dropouts2, num_dropouts3]
                    
        
                    # add dense layer
                    model.add(GaussianNoise(noise))
                    model.add(Dense(l1[i],
                                    activation=activation,
                                    name=name, kernel_regularizer=regularizers.l1_l2(l1=regularization1,l2=regularization2)))
                    model.add(tf.keras.layers.Dropout(l2[i]))

                # use softmax-activation for classification.
                model.add(Dense(3, activation='softmax'))

                # Use the Adam method for training the network.
                optimizer = keras.optimizers.SGD(lr=learning_rate)

                # compile the model so it can be trained.
                model.compile(optimizer=optimizer,
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])

                return model


            @use_named_args(dimensions=dimensions)
            def fitness(learning_rate, activation, regularization1,regularization2,noise, num_dense_nodes1, num_dense_nodes2, num_dense_nodes3,
                        num_dropouts1, num_dropouts2, num_dropouts3, num_batch_size):
                model = create_model(learning_rate=learning_rate,
                                     activation=activation,
                                     regularization1=regularization1,
                                     regularization2=regularization2,
                                     noise=noise,
                                     num_dense_nodes1=num_dense_nodes1,
                                     num_dense_nodes2=num_dense_nodes2,
                                     num_dense_nodes3=num_dense_nodes3,
                                     num_dropouts1=num_dropouts1, num_dropouts2=num_dropouts2,
                                     num_dropouts3=num_dropouts3)
                callback_log = TensorBoard(
                    #      log_dir=log_dir,
                    histogram_freq=0,
                    write_graph=True,
                    write_grads=False,
                    write_images=False)
                es = EarlyStopping(monitor='val_accuracy', patience=10)
                # Use Keras to train the model.
                history = model.fit(x=X_train,
                                    y=y_train,
                                    epochs=200,
                                    batch_size=num_batch_size,
                                    validation_data=(X_test, y_test),
                                    callbacks=[callback_log], class_weight=class_weights)
                accuracy = history.history['val_accuracy'][-1]
                print()
                print("Accuracy: {0:.2%}".format(accuracy))
                print()
                global best_accuracy

                if accuracy > best_accuracy:
                    # Save the new model to harddisk.
                    model.save('home/mat/Desktop/trials')
                    out.append([learning_rate, num_dense_nodes1, lay, activation, regularization1,regularization2,noise,
                                num_dense_nodes2,
                                num_dense_nodes3, num_dropouts1, num_dropouts2, num_dropouts3,
                                'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none',
                                'none', num_batch_size, accuracy])
                    best_accuracy = accuracy
                del model
                K.clear_session()
                return -accuracy


            default_parameters = def_para[lay]
            print(default_parameters)
            fitness(x=default_parameters)

            try:
                search_result = gp_minimize(func=fitness,
                                            dimensions=dimensions,
                                            acq_func='EI',  # Expected Improvement.
                                            n_calls=12,
                                            random_state=1234)
            except ValueError:
                pass
        elif lay == 2:
            def create_model(learning_rate, activation, regularization1,regularization2,noise, num_dense_nodes1, num_dropouts1, num_dense_nodes2,
                             num_dropouts2):
                model = Sequential()

                model.add(InputLayer(input_shape=(n_features,)))

                for i in range(lay):
                    name = 'layer_dense_{0}'.format(i + 1)
                    l1 = [num_dense_nodes1, num_dense_nodes2]
                    l2 = [num_dropouts1, num_dropouts2]
                    # add dense layer
                    model.add(GaussianNoise(noise))
                    model.add(Dense(l1[i],
                                    activation=activation,
                                    name=name, kernel_regularizer=regularizers.l1_l2(l1=regularization1,l2=regularization2)))
                    model.add(tf.keras.layers.Dropout(l2[i]))

                # use softmax-activation for classification.
                model.add(Dense(3, activation='softmax'))

                # Use the Adam method for training the network.
                optimizer = keras.optimizers.SGD(lr=learning_rate)

                # compile the model so it can be trained.
                model.compile(optimizer=optimizer,
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])

                return model


            @use_named_args(dimensions=dimensions)
            def fitness(learning_rate, activation, regularization1,regularization2,noise, num_batch_size, num_dense_nodes1, num_dropouts1,
                        num_dense_nodes2, num_dropouts2):
                model = create_model(learning_rate=learning_rate,
                                     activation=activation,
                                     regularization1=regularization1,
                                     regularization2=regularization2,
                                     noise=noise,
                                     num_dense_nodes1=num_dense_nodes1,
                                     num_dense_nodes2=num_dense_nodes2,
                                     num_dropouts1=num_dropouts1,
                                     num_dropouts2=num_dropouts2)
                callback_log = TensorBoard(
                    #      log_dir=log_dir,
                    histogram_freq=0,
                    write_graph=True,
                    write_grads=False,
                    write_images=False)
                es = EarlyStopping(monitor='val_accuracy', patience=10)
                # Use Keras to train the model.
                history = model.fit(x=X_train,
                                    y=y_train,
                                    epochs=200,
                                    batch_size=num_batch_size,
                                    validation_data=(X_test, y_test),
                                    callbacks=[callback_log], class_weight=class_weights)
                accuracy = history.history['val_accuracy'][-1]
                print()
                print("Accuracy: {0:.2%}".format(accuracy))
                print()
                global best_accuracy
                best_accuracy=0.4
                if accuracy > best_accuracy:
                    # Save the new model to harddisk.
                    model.save('home/mat/Desktop/trials')
                    out.append([learning_rate, num_dense_nodes1, lay, activation, regularization1,regularization2,noise,
                                num_dense_nodes2, num_dropouts1, num_dropouts2, 'none', 'none', 'none', 'none', 'none',
                                'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', num_batch_size, accuracy])
                    best_accuracy = accuracy
                del model
                K.clear_session()
                return -accuracy


            default_parameters = def_para[lay]
            fitness(x=default_parameters)
            try:
                search_result = gp_minimize(func=fitness,
                                            dimensions=dimensions,
                                            acq_func='EI',  # Expected Improvement.
                                            n_calls=12,
                                            random_state=1234)
            except ValueError:
                pass
        
        
        elif lay == 1:
            def create_model(learning_rate, activation, regularization1,regularization2,noise, num_dense_nodes1, num_dropouts1):
                model = Sequential()

                model.add(InputLayer(input_shape=(n_features,)))

                for i in range(lay):
                    name = 'layer_dense_{0}'.format(i + 1)
                    l1 = [num_dense_nodes1]
                    l2 = [num_dropouts1]
                    
        
                    # add dense layer
                    model.add(GaussianNoise(noise))
                    model.add(Dense(l1[i],
                                    activation=activation,
                                    name=name, kernel_regularizer=regularizers.l1_l2(l1=regularization1,l2=regularization2)))
                    model.add(tf.keras.layers.Dropout(l2[i]))

                # use softmax-activation for classification.
                model.add(Dense(3, activation='softmax'))

                # Use the Adam method for training the network.
                optimizer = keras.optimizers.SGD(lr=learning_rate)

                # compile the model so it can be trained.
                model.compile(optimizer=optimizer,
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])

                return model


            @use_named_args(dimensions=dimensions)
            def fitness(learning_rate, activation, regularization1,regularization2,noise, num_batch_size, num_dense_nodes1, num_dropouts1):
                model = create_model(learning_rate=learning_rate,
                                     activation=activation,
                                     regularization1=regularization1,
                                     regularization2=regularization2,
                                     noise=noise,
                                     num_dense_nodes1=num_dense_nodes1,
                                     num_dropouts1=num_dropouts1)
                callback_log = TensorBoard(
                    #      log_dir=log_dir,
                    histogram_freq=0,
                    write_graph=True,
                    write_grads=False,
                    write_images=False)
                es = EarlyStopping(monitor='val_accuracy', patience=10)
                # Use Keras to train the model.
                history = model.fit(x=X_train,
                                    y=y_train,
                                    epochs=200,
                                    batch_size=num_batch_size,
                                    validation_data=(X_test, y_test),
                                    callbacks=[callback_log], class_weight=class_weights)
                accuracy = history.history['val_accuracy'][-1]
                print()
                print("Accuracy: {0:.2%}".format(accuracy))
                print()
                global best_accuracy
                best_accuracy=0.4
                if accuracy > best_accuracy:
                    # Save the new model to harddisk.
                    model.save('home/mat/Desktop/trials')
                    out.append([learning_rate, num_dense_nodes1, lay, activation, regularization1,regularization2,noise,
                                num_dropouts1, 'none', 'none', 'none', 'none', 'none',
                                'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', num_batch_size, accuracy])
                    best_accuracy = accuracy
                del model
                K.clear_session()
                return -accuracy


            default_parameters = def_para[lay]
            fitness(x=default_parameters)
            try:
                search_result = gp_minimize(func=fitness,
                                            dimensions=dimensions,
                                            acq_func='EI',  # Expected Improvement.
                                            n_calls=12,
                                            random_state=1234)
            except ValueError:
                pass



with open('/home/mat/Desktop/cond_coop/trials/NeuralNets/20210429_sesca3_l1_l2.csv', "w", newline="") as f:
    writer = csv.writer(f)
    for i in out:
        writer.writerow(i)

print(out)

end_date = datetime.now()
end_date = end_date.strftime("%d/%m/%Y %H:%M:%S")
print(start_date)
print(end_date)
print("session started", start_date, "and ended", end_date)