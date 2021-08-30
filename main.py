import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import DataPreProcessor
import sys
import os
import csv
import tensorflow
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM

path = "Data"
directory = os.fsencode(path)
dirs = os.listdir(path)

Matcher=pd.read_csv("Symbol_Piotroski1.csv")

Ticker=list(Matcher['SYMBOL'])

result = []

for file in dirs:
    if file.split('.')[0] in Ticker:

        np.random.seed(7)

        current_file = "Data/" + str(file)
        dataset = pd.read_csv(current_file, usecols=[1,2,3,4])
        dataset = dataset.reindex(index = dataset.index[::-1])

        obsolete = np.arange(1, len(dataset) +1, 1)

        OHLC_avg = dataset.mean(axis=1)
        OHLC_avg_copy = dataset.mean(axis=1)
        HLC_avg = dataset[['High', 'Low', 'Close']].mean(axis=1)
        close_val = dataset[['Close']]

        plt.plot(obsolete, OHLC_avg, 'r', label='OHLC_avg')
        plt.plot(obsolete, HLC_avg, 'b', label='HLC_avg')
        plt.plot(obsolete, close_val, 'g', label='Closing Price')
        plt.legend(loc = 'upper right')
        plt.show()

        OHLC_avg = np.reshape(OHLC_avg.values, (len(OHLC_avg),1))
        scaler = MinMaxScaler(feature_range=(0,1))
        OHLC_avg = scaler.fit_transform(OHLC_avg)

        train_OHLC = int(len(OHLC_avg) * .75)
        test_OHLC = len(OHLC_avg) - train_OHLC
        train_OHLC, test_OHLC = OHLC_avg[0:train_OHLC,:], OHLC_avg[train_OHLC:len(OHLC_avg),:]

        trainX, trainY = DataPreProcessor.new_dataset(train_OHLC,5)
        testX, testY = DataPreProcessor.new_dataset(test_OHLC, 5)

        trainX = np.reshape(trainX, (trainX.shape[0], 1,trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        step_size = 5

        model = Sequential()
        model.add(LSTM(32, input_shape=(1, step_size), return_sequences=True))
        model.add(LSTM(16))
        model.add(Dense(1))
        model.add(Activation('linear'))

        model.compile(loss='mean_squared_error', optimizer='adagrad')
        model.fit(trainX,trainY,epochs=50, batch_size=15, verbose=2)

        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        # DE-NORMALIZING FOR PLOTTING

        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([trainY])
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY])

        # TRAINING RMSE
        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
        print('Train RMSE: %.2f' % (trainScore))

        # TEST RMSE
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
        print('Test RMSE: %.2f' % (testScore))

        # CREATING SIMILAR DATASET TO PLOT TRAINING PREDICTIONS
        trainPredictPlot = np.empty_like(OHLC_avg)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[step_size:len(trainPredict) + step_size, :] = trainPredict

        # CREATING SIMILAR DATASSET TO PLOT TEST PREDICTIONS
        testPredictPlot = np.empty_like(OHLC_avg)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict) + (step_size * 2) + 1:len(OHLC_avg) - 1, :] = testPredict

        # DE-NORMALIZING MAIN DATASET
        OHLC_avg = scaler.inverse_transform(OHLC_avg)

        # PLOT OF MAIN OHLC VALUES, TRAIN PREDICTIONS AND TEST PREDICTIONS
        plt.plot(OHLC_avg, 'g', label='original dataset')
        plt.plot(trainPredictPlot, 'r', label='training set')
        plt.plot(testPredictPlot, 'b', label='predicted stock price/test set')
        plt.legend(loc='upper right')
        plt.xlabel('Time in Days')
        plt.ylabel('OHLC Value of Apple Stocks')
        plt.show()

        # PREDICT FUTURE VALUES
        last_val = OHLC_avg[np.array([-1, -2, -3, -4, -5])]
        last_val = scaler.fit_transform(last_val)
        # last_val_scaled = last_val/last_val
        # next_val = model.predict(np.reshape(last_val, (1,1,step_size)))
        # print ("Last Day Value:", np.asscalar(last_val))
        # print ("Next Day Value:", np.asscalar(last_val*next_val))

        pred_vals = []
        pred_vals1 = []
        pred_vals1.append(file)
        for i in range(0, 5):
            # last_val_scaled = last_val/last_val
            print(last_val)
            next_val = model.predict(np.reshape(last_val, (1, 1, step_size)))
            pred_vals.append(next_val)
            print(next_val)
            last_val = np.append(last_val, next_val)
            last_val = np.delete(last_val, 0)

            # next_vals.append(np.asscalar(model.predict(np.reshape(, (1,1,step_size)))))
            # last_val1.append(next_vals[i-1]*last_val1[i])
        # pred_vals=scaler.inverse_transform(np.array(pred_vals).reshape(1,5))

        ### Scaling Values back using last 5 values as scale standard
        pred_vals = np.array(pred_vals).reshape(1, 5)
        last_val_unscaled = np.array(OHLC_avg_copy[np.array([0, 1, 2, 3, 4])]).reshape(1, 5)

        scaler = MinMaxScaler(feature_range=(0, 1))
        last_val_scaler = scaler.fit_transform(last_val_unscaled)

        pred_vals_rescaled = scaler.inverse_transform(pred_vals)

        a = list(pred_vals_rescaled)[0]
        pred_vals1.append(a)

        result.append(pred_vals1)

    res = pd.DataFrame(result)
    res.to_csv('results.csv', index=False, header=False)










