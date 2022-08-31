import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from keras.layers import Flatten
from keras.models import load_model
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from csv import reader
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

import numpy as np
import keras.layers
from keras.models import model_from_json
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from sklearn.preprocessing import MinMaxScaler
import random
import dbConstruct
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from keras.wrappers.scikit_learn import KerasRegressor

# load dataset
# open file in read mode
#openpalm 0
#fist 1
#point 2
#twopoint 3

class floatbind:
    def __init__(self):
        pass
    #save model
    def save_model(model):
        # saving model
        json_model = model.to_json()
        open('curieNet_model/model_architecture.json', 'w').write(json_model)
        # saving weights
        model.save_weights('curieNet_model/model_weights.h5', overwrite=True)
    #load model from saved model
    def load_model():
        # loading model
        model = model_from_json(open('curieNet_model/model_architecture.json').read())
        model.load_weights('curieNet_model/model_weights.h5')
        model.compile(loss='mse', optimizer='sgd')
        return model
    #load data from h5 file
    def load_curie_h5():
        h5file, table = dbConstruct.openLog("curieDB")
        names,tg = dbConstruct.rowDump(table)
        dbConstruct.closeLog(h5file, table)
        tg = np.array(tg)
        #test[:, 0]
        bic = pd.DataFrame({"temperature": tg[:, 3],
                            "pressure": tg[:, 2],
                            "pir": tg[:, 1],
                            "light": tg[:, 0],
                            "time": tg[:, 4]})
        return(bic)

    #model architecture
    def baseline_model():
        model = Sequential()
        model.add(keras.Input(shape=(1,)))
        model.add(Dense(164, activation='linear'))
        model.add(Dense(164, activation='linear'))
        model.add(Dense(164, activation='linear'))
        model.add(Dense(164, activation='linear'))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(164, activation='linear'))
        model.add(Dense(164, activation='linear'))
        model.add(Dense(164, activation='linear'))
        model.add(Dense(164, activation='linear'))
        model.add(Dropout(0.2))
        model.add(Dense(21, activation='linear'))
        model.add(Dense(21, activation='linear'))
        model.add(Dense(21, activation='linear'))
        model.add(Dense(21, activation='linear'))
        model.add(Dense(21, activation='linear'))
        model.add(Dense(21, activation='linear'))
        model.add(Dense(21, activation='linear'))
        model.add(Dense(21, activation='relu'))
        #model.add(Dense(100, activation='softmax'))
        model.add(Dense(1, activation='linear'))
        model.add(Dense(1, activation='linear'))
        model.add(Dense(1, activation='linear'))
        model.add(Dense(1, activation='linear'))
    	# Compile model

        model.compile(loss='mse',  optimizer='sgd',metrics=['mean_absolute_error'])
        return model

    #train and save model
    def train_network(vi, vo):
        database = load_curie_h5()
        variablei = 'pressure'
        variableo = 'temperature'
        for v in range(0, len( database[variablei])):
            database[variablei][v] =float(database[variablei][v])
        for v in range(0, len( database[variableo])):
            database[variableo][v] =float(database[variableo][v])
        X = np.asarray(database[variablei].to_list())
        Y = np.asarray(database[variableo].to_list())
        X = X.reshape(X.shape[0], 1)
        Y = Y.reshape(Y.shape[0], 1)
        print(X)
        print(Y)
        scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
        scalarX.fit(X)
        scalarY.fit(Y)
        X = scalarX.transform(X)
        Y = scalarY.transform(Y)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
        print(X_train.shape)
        print(Y_train.shape)
        #estimator = KerasRegressor(build_fn=baseline_model, epochs=200, batch_size=8, verbose=1)
        model = baseline_model()
        model.fit(X_train, Y_train, epochs=200, batch_size=8, verbose=1)
        save_model(model)
        return(model, scalarX, scalarY, X_train, X_test, Y_train, Y_test)

    #graph predictions versus actual values
    def graph_predictions(model, scalarX, scalarY, X_train, X_test, Y_train, Y_test):
        predictions = model.predict(X_test, verbose=0)
        for x in range(0, len(predictions)):
            print("Test input: ", str(scalarX.inverse_transform(X_test[x].reshape(1, -1))))
            print(" Actual: ", str(scalarY.inverse_transform(Y_test[x].reshape(1, -1))))
            print(" Predicted: "+str(scalarY.inverse_transform(predictions[x].reshape(1, -1))))
            xs = scalarX.inverse_transform(X_test[x].reshape(1, -1))
            ya = scalarY.inverse_transform(Y_test[x].reshape(1, -1))
            yp = scalarY.inverse_transform(predictions[x].reshape(1, -1))
            sns.scatterplot(x=xs[0], y=ya[0], color = "blue")
            sns.scatterplot(x=xs[0], y=yp[0], color = "red")
        plt.show()

    #generator for live usage of model
    def pred_gen():
        model = load_model()
        while(True):
            val = yield
            #print("  ")
            if val is not None:
                if(len(val)==1):
                    yield([np.argmax(model.predict(np.array([val[0],])), axis=-1)])
                elif len(val)==2 :
                    yield([np.argmax(model.predict(np.array([val[0],val[1]])), axis=-1)])
    def export_to_tflite():
        model = load_model()
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()


m, scalarX, scalarY, X_train, X_test, Y_train, Y_test = train_network("i","o")
graph_predictions(m, scalarX, scalarY, X_train, X_test, Y_train, Y_test)
