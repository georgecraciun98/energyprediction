 # -*- coding: utf-8 -*-
import os
from math import sqrt
from pandas import read_csv
from numpy import split
from numpy import array
from sklearn.metrics import mean_squared_error,mean_absolute_error
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf


from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from matplotlib import pyplot

def split_dataset(values):
    train,test = values[1:-328],values[-328:-6]
    #Restructure into weekly data
    train=array(split(train,len(train)/7))
    test=array(split(test,len(test)/7))
    return train,test


#Evaluation metric

"""
    actual represent actual results,predicted 
    represents predicted results
    """
def evaluate_forecasts(actual,predicted):
    scores_rmse = list()
    scores_mae= list()
	# calculate an RMSE score for each day
    for i in range(actual.shape[1]):
		# calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])

        mae= mean_absolute_error(actual[:, i], predicted[:, i])
		# calculate rmse
        rmse = sqrt(mse)
		# store
        scores_rmse.append(rmse)
        scores_mae.append(mae)
	# calculate overall RMSE
    s_rmse = 0
    s_mae = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s_rmse += (actual[row, col] - predicted[row, col])**2
            s_mae += abs((actual[row, col] - predicted[row, col]))
    score_rmse = sqrt(s_rmse / (actual.shape[0] * actual.shape[1]))
    score_mae = s_mae / (actual.shape[0] * actual.shape[1])
    return score_rmse, scores_rmse, score_mae, scores_mae

def summarize_scores(name,score,scores):
    s_scores= ', '.join(['%.1f' % s for s in scores])
    print('Scores')
    print('%s: [%.3f] %s' % (name, score, s_scores))
    
    
def to_supervised(train,n_input,n_out=7):
    """
    convert history into inputs and outputs
    """
    data=train.reshape((train.shape[0]*train.shape[1],train.shape[2]))
    
    x,y = list(),list()
    in_start=0
    
    #Step over the entire history one time step at a time
    for _ in range(len(data)):
        #Define the end of the input sequence
        in_end=in_start+n_input
        out_end=in_end+n_out
        
        #Check if we have enough data 
        if out_end < len(data):
            x_input=data[in_start:in_end,0]
            x_input=x_input.reshape((len(x_input),1))
            
            x.append(x_input)
            y.append(data[in_end:out_end,0])
        
        in_start += 1

    return array(x),array(y)

def build_model(train,n_input):
    
    train_x,train_y=to_supervised(train, n_input)
    #define working params
    
    verbose,epochs,batch_size=0,70,16    

    n_timesteps,n_features,n_outputs=train_x.shape[1],train_x.shape[2],train_y.shape[1]

    
    
    
    model = keras.Sequential()
    model.add(LSTM(200,activation='relu',input_shape=(n_timesteps,n_features)))
    
    model.add(Dense(100,activation='relu'))
    model.add(Dense(n_outputs))
    
    model.compile(loss='mse', optimizer='adam')
    
    model.fit(train_x,train_y,epochs=epochs,batch_size=batch_size,verbose=verbose)
    
    return model
def forecast(model, history, n_input):
    """
    The forecast() function below implements this and takes 
    as arguments the model fit on the training dataset,
    the history of data observed so far, and the number 
    of input time steps expected by the model.
    """
    data = array(history)

    data= data.reshape((data.shape[0]*data.shape[1],data.shape[2]))
    
    #retrive the last observations for input data
    input_x = data[-n_input:, 0]
   
    input_x = input_x.reshape((1,len(input_x),1))
    #forecast the next week
    
    yhat= model.predict(input_x,verbose=0)
    yhat=yhat[0]
    return yhat
    

#n_input used for define the number of prior observations

def evaluate_model(train,test,n_input):
    model =build_model(train,n_input)
    
    history=[x for x in train]
    
    predictions=list()
    
    for x in range(len(test)):
        week_sequence=forecast(model,history,n_input)
        
        predictions.append(week_sequence)
        
        history.append(test[x,:])
    
    predictions = array(predictions)
    
    score_rmse, scores_rmse, score_mae, scores_mae=evaluate_forecasts(test[:,:,0], predictions)
        
        
    
    return score_rmse, scores_rmse, score_mae, scores_mae
    

    


    
    



