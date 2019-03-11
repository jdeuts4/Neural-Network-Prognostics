import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation

#embedding dimension (https://stackoverflow.com/questions/45645511/rolling-window-in-python)
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    
"""
example of above rolling_window(a,2) 


a = np.array([1, 2, 3, 4, 5, 6])
rolling_window(a,2) returns


array([[1, 2],
       [2, 3],
       [3, 4],
       [4, 5],
       [5, 6]])
"""


# GENERATE DATA


t = np.arange(0,1000) # t = time from t=0 to 1000


#after processing your vibration data your signal
# you may have a 1d signal that looks like this




np.random.seed(1) #set seed for reproducibility 
signal = .01*t**1.5 + np.random.normal(0,10,1000)+50


# END GENERATE DATA




# get x,y for supervised learning
d = 2 #embedding dimension to embed/window signals
x = rolling_window(signal,d)
y = range(0,len(x))
y.reverse() # RUL






# function from keras that builds the neural network


def buildModel(d):
    model = Sequential()
    
    #first layer with 50 hidden units
    model.add(Dense(50,input_dim=d))
    model.add(Activation('relu'))
    
    
    #second layer with 20 hidden units
    model.add(Dense(20))
    model.add(Activation('relu'))


    #output layer with just a single output which outups the RUL
    model.add(Dense(1))


    model.compile(loss='mse', optimizer='adam')
    return model


#create the network, run it for 200 epcohs and predict the results
model = buildModel(d)
model.fit(x,y,nb_epoch=200)
predictions = model.predict(x).flatten()    
    


    
    
#plot the predicted rul vs the true rul    
plt.figure(figsize=(7,7))
plt.plot(y,'g')
plt.plot(predictions,'r')
plt.xlabel('t')
plt.ylabel('Remaining Useful Life')
plt.legend(['True RUL','Estimated RUL'])
