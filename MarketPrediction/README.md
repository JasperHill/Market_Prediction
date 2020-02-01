# BPI_RNN

A recurrent neural network for predicting the Bitcoin Price Index (BPI)

The network takes market data from a csv file in the subdirectory, data, in the current working directory. All-time highs and lows are extraced and processed according
to parameters, which must currently be set within the script file. The network is trained on the first and second discrete derivatives of the BPI with the goal of
producing future first derivatives. The dataset from which the inputs and outputs originate also contains absolute prices so that the trained network output can be
integrated with the final high and low prices to generate a forecast.
