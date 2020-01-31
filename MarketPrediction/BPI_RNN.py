#########################################################################
##  BPI_RNN.py
##  Jan. 2020 - J. Hill
#########################################################################

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pathlib
import tensorflow        as tf
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM, Dense, InputLayer, Reshape
from tensorflow.keras.models import Sequential

"""
A recurrent neural network featuring a long-short-term memory layer. The model takes first and second derivatives
as its input and produces first derivatives. An array of high and low absolute prices are stored in the dataset
as well so that the output derivatives can be integrated with the final high/low input prices as the starting points.
"""

#########################################################################
##  preprocess data
#########################################################################

## import data file
cwd = os.getcwd()
data_path = os.path.join(cwd,'data')
data_path = os.path.join(data_path,'BTC_USD_2013-10-01_2020-01-28-CoinDesk.csv')

pd_object = pd.read_csv(data_path,usecols=['Date','24h High (USD)','24h Low (USD)'])
np_object = np.asarray(pd_object)

## separate dates from np_object and compute discrete first and second derivatives
dates          =             np_object[:,0]
prices_prime   =            np_object[:,1:]
dprices_prime  = np.diff(prices_prime,n=1,axis=0)
ddprices_prime = np.diff(prices_prime,n=2,axis=0)

## trim arrays to identical lengths
ddprices =       ddprices_prime
dprices  =        dprices_prime[1:,:]
prices   =         prices_prime[2:,:]
dates    =                dates[2:]

## set dimensions of the input data
train_frac  =                        0.5
OFFSET      =                          0
LENGTH      =                len(prices)
START_IDX   =                          0
TRAIN_SPLIT = np.ceil(train_frac*LENGTH)
HIST_SIZE   =                         20
TARG_SIZE   =                         10

BATCH_SIZE  =                        512
BUFFER_SIZE =                       1000

dataset  = np.asarray([prices,dprices,ddprices])
## permute the axes so that data has shape (LENGTH,3,2) corresponding to  (timestep, derivative, high/low)
dataset  = np.swapaxes(dataset,0,1)

## set the time span of each input element and label
## create input elements and labels for each index between START_INDEX AND END_INDEX
def format_ds(ds,start_index,end_index,history_size,target_size):
    ref_prices = []
    data       = []
    labels     = []    
    
    start_index += history_size

    if end_index is None: end_index = len(ds) - target_size

    for i in range(start_index, end_index):
        idxs = range(i-history_size, i)        
        ## the data only includes 1st and 2nd derivatives of highs and lows
        data.append(np.reshape(ds[idxs,1:,:], (history_size,2,2)))

        idxs = range(i,i+target_size)
        ## the reference prices are included in the dataset so that the output derivatives can be integrated
        ref_prices.append(np.reshape(ds[idxs,0,:], (target_size,2)))
        
        ## reformat the labels to include only the 1st derivative
        labels.append(np.reshape(ds[idxs,1,:], (target_size,2)))

        
    return np.array(ref_prices, dtype=np.float32), np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32)

train_prices, x_train, y_train = format_ds(dataset, START_IDX, int(TRAIN_SPLIT), HIST_SIZE, TARG_SIZE)
test_prices, x_test, y_test   = format_ds(dataset, int(TRAIN_SPLIT), None, HIST_SIZE, TARG_SIZE)

## auxiliary function for converting derivative outputs to market prices
def integrate_outputs(ref_prices, y_pred):
    prices = ref_prices[0]
    y_int = np.empty(y_pred.shape)
    
    for i in range(y_pred.shape[0]):        
        prices += y_pred[i]
        y_int[i] = prices

    return y_int
        

## auxiliary plotting function for visualization
def plot_data(arrays, delta, title, filename):
    fig = plt.figure(figsize=(10,10))
    labels = [['True high', 'True low'], ['Predicted high', 'Predicted low']]
    colors = [['blue','red'], ['green','yellow']]
    
    past = arrays[0].shape[0]
    time_steps = list(range(-past,0))
    future = list(range(delta))

    plt.title(title)

    for i,x in enumerate(arrays):
        if i:
            ## array indices are (list index,time_step,high/low)
            plt.plot(future, arrays[i][:,0], color=colors[i][0], marker='.', markersize=1, label=labels[i][0])
            plt.plot(future, arrays[i][:,1], color=colors[i][1], marker='.', markersize=1, label=labels[i][1])

        else:
            plt.plot(future, arrays[i][:,0], color=colors[i][0], marker='.', markersize=1, label=labels[i][0])
            plt.plot(future, arrays[i][:,1], color=colors[i][1], marker='.', markersize=1, label=labels[i][1])

    plt.legend()
    plt.xlim(xmin=time_steps[0], xmax=(delta+5)*2)
    plt.xlabel('time step (d)')
    plt.savefig(filename+'.pdf')        
    plt.close(fig)
    
#plot_data([y_train[0]], delta=0, title='Sample_Window', filename='Sample_Window')


#########################################################################
##  create tensorflow dataset and define training parameters
#########################################################################

train_ds =  tf.data.Dataset.from_tensor_slices((train_prices, x_train, y_train))
train_ds =      train_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

test_ds =      tf.data.Dataset.from_tensor_slices((test_prices, x_test, y_test))
test_ds =                                     test_ds.batch(BATCH_SIZE).repeat()

STEPS_PER_EPOCH  =                          int(np.ceil(TRAIN_SPLIT/BATCH_SIZE))
VALIDATION_STEPS =                 int(np.ceil((LENGTH-TRAIN_SPLIT)/BATCH_SIZE))
EPOCHS           =                                                   range(2000)

#########################################################################
##  define training and testing steps
#########################################################################

train_MSE_hist = []
test_MSE_hist  = []

optimizer   = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.MeanSquaredError()
train_MSE   = tf.keras.metrics.MeanSquaredError()
test_MSE    = tf.keras.metrics.MeanSquaredError()

@tf.function
def train_step(ds, model, steps):
    for step in steps:
        train_prices, x_train, y_train = next(iter(ds))
        
        with tf.GradientTape() as tape:
            y_pred = model(x_train, training=True)

            loss = loss_object(y_train, y_pred)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
        train_MSE.update_state(y_train, y_pred)
    train_MSE_hist.append(train_MSE.result())

def test_step(ds, model, steps):
    for step in steps:
        test_prices, x_test, y_test = next(iter(ds))
        y_pred = model(x_test, training=False)

        test_MSE.update_state(y_test, y_pred)
    test_MSE_hist.append(test_MSE.result())

def print_dots(epoch):
    if (epoch%1 == 0):  print('.',end='')
    if (epoch%10 == 0): print('.')


#########################################################################
##  build and train the model
#########################################################################
    
simple_lstm_model = Sequential([InputLayer(input_shape=x_train.shape[1:]),
                                Reshape((x_train.shape[1],4)), ## collapse the derivative and high/low axes
                                LSTM(int(10), activation='tanh'),
                                Dense(int(2*TARG_SIZE), activation=None),
                                Reshape((TARG_SIZE,2))])

simple_lstm_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MSE, metrics=[tf.keras.metrics.MSE])
simple_lstm_model.summary()


VALIDATION_STEPS = STEPS_PER_EPOCH
for epoch in EPOCHS:
    train_MSE.reset_states()
    test_MSE.reset_states()

    train_step(train_ds, simple_lstm_model, range(STEPS_PER_EPOCH))
    test_step(test_ds, simple_lstm_model, range(VALIDATION_STEPS))

    if (epoch%10 == 0):
        if (epoch <= 100 or epoch%100 == 0):
            print ('epoch {} | train_MSE: {} | test_MSE: {}'.format(epoch, train_MSE.result(), test_MSE.result()))
            

print('')

for rp,x,y in test_ds.take(1):
    predictions = simple_lstm_model(x, training=False)
    y_int = integrate_outputs(rp[0],predictions[0])
    plot_data([rp[0].numpy(), y_int], TARG_SIZE, title='{}-Epoch Test Results'.format(EPOCHS[-1]+1), filename='{}Epoch_Test_Results'.format(EPOCHS[-1]+1))
