#http://archive.ics.uci.edu/ml/machine-learning-databases/00235/
#https://www.kaggle.com/amirrezaeian/time-series-data-analysis-using-lstm-tutorial

import sys
import numpy as np # linear algebra
from scipy.stats import randint
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns # used for plot interactive graph.
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.preprocessing import StandardScaler # for normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline # pipeline making
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error,r2_score

## for Deep-learing:
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout

df = pd.read_csv('C:/Users/BCINAR-PC.DESKTOP-ML48OK9/Desktop/works/Pycharm/machinelearning/household_power_consumption.txt', sep=';',
                 parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True,
                 low_memory=False, na_values=['nan','?'], index_col='dt')

print(df.head())
print("-------")
print(df.info())
print("---------")
print(df.dtypes)
print("---------")
print(df.shape)
print("---------")
print(df.describe())
print("--------")
print(df.columns)
print("--------")

print(df.isnull().sum())

droping_list_all=[]
for j in range(0,7):
    if not df.iloc[:, j].notnull().all():
        droping_list_all.append(j)

for j in range(0,7):
        df.iloc[:,j]=df.iloc[:,j].fillna(df.iloc[:,j].mean())

print(df.isnull().sum())


# df.Global_active_power.resample('D').sum().plot(title='Global_active_power resampled over day for sum')
# #df.Global_active_power.resample('D').mean().plot(title='Global_active_power resampled over day', color='red')
# plt.tight_layout()
# plt.show()

# df.Global_active_power.resample('D').mean().plot(title='Global_active_power resampled over day for mean', color='red')
# plt.tight_layout()
# plt.show()

# r = df.Global_intensity.resample('D').agg(['mean', 'std'])
# r.plot(subplots = True, title='Global_intensity resampled over day')
# plt.show()

# r2 = df.Global_reactive_power.resample('D').agg(['mean', 'std'])
# r2.plot(subplots = True, title='Global_reactive_power resampled over day', color='red')
# plt.show()

# df['Global_active_power'].resample('M').mean().plot(kind='bar')
# plt.xticks(rotation=60)
# plt.ylabel('Global_active_power')
# plt.title('Global_active_power per month (averaged over month)')
# plt.show()

# df['Global_active_power'].resample('Q').mean().plot(kind='bar')
# plt.xticks(rotation=60)
# plt.ylabel('Global_active_power')
# plt.title('Global_active_power per quarter (averaged over quarter)')
# plt.show()

# ## mean of 'Voltage' resampled over month
# df['Voltage'].resample('M').mean().plot(kind='bar', color='red')
# plt.xticks(rotation=60)
# plt.ylabel('Voltage')
# plt.title('Voltage per quarter (summed over quarter)')
# plt.show()

# df['Sub_metering_1'].resample('M').mean().plot(kind='bar', color='brown')
# plt.xticks(rotation=60)
# plt.ylabel('Sub_metering_1')
# plt.title('Sub_metering_1 per quarter (summed over quarter)')
# plt.show()

# Below I compare the mean of different featuresresampled over day.
# specify columns to plot
# cols = [0, 1, 2, 3, 5, 6]
# i = 1
# groups=cols
# values = df.resample('D').mean().values
# # plot each column
# plt.figure(figsize=(15, 10))
# for group in groups:
# 	plt.subplot(len(cols), 1, i)
# 	plt.plot(values[:, group])
# 	plt.title(df.columns[group], y=0.75, loc='right')
# 	i += 1
# plt.show()

# print(df.Global_reactive_power)

# df.Global_reactive_power.resample('W').mean().plot(color='y', legend=True)
# df.Global_active_power.resample('W').mean().plot(color='r', legend=True)
# df.Sub_metering_1.resample('W').mean().plot(color='b', legend=True)
# df.Global_intensity.resample('W').mean().plot(color='g', legend=True)
# plt.show()

# Below I show hist plot of the mean of different feature resampled over month
# df.Global_active_power.resample('M').mean().plot(kind='hist', color='r', legend=True )
# df.Global_reactive_power.resample('M').mean().plot(kind='hist',color='b', legend=True)
# #df.Voltage.resample('M').sum().plot(kind='hist',color='g', legend=True)
# df.Global_intensity.resample('M').mean().plot(kind='hist', color='g', legend=True)
# df.Sub_metering_1.resample('M').mean().plot(kind='hist', color='y', legend=True)
# plt.show()


# data_returns = df.pct_change()
# sns.jointplot(x='Global_reactive_power', y='Global_active_power', data=data_returns)
# plt.show()
# sns.jointplot(x='Voltage', y='Global_active_power', data=data_returns)
# plt.show()

# plt.matshow(df.corr(method='spearman'),vmax=1,vmin=-1,cmap='PRGn')
# plt.title('without resampling', size=15)
# plt.colorbar()
# plt.show()

plt.matshow(df.resample('M').mean().corr(method='spearman'),vmax=1,vmin=-1,cmap='PRGn')
plt.title('resampled over month', size=15)
plt.colorbar()
plt.margins(0.02)
plt.matshow(df.resample('A').mean().corr(method='spearman'),vmax=1,vmin=-1,cmap='PRGn')
plt.title('resampled over year', size=15)
plt.colorbar()
plt.show()

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	dff = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(dff.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(dff.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

df_resample = df.resample('h').mean()
print(df_resample.shape)

## If you would like to train based on the resampled data (over hour), then used below
values = df_resample.values


## full data without resampling
#values = df.values

# integer encode direction
# ensure all data is float
#values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)

# drop columns we don't want to predict
reframed.drop(reframed.columns[[8,9,10,11,12,13]], axis=1, inplace=True)
print(reframed.head())

# split into train and test sets
values = reframed.values

n_train_time = 365*24
train = values[:n_train_time, :]
test = values[n_train_time:, :]
##test = values[n_train_time:n_test_time, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
# We reshaped the input into the 3D format as expected by LSTMs, namely [samples, timesteps, features].

model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))
#    model.add(LSTM(70))
#    model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=19, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], 7))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, -6:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, -6:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

aa=[x for x in range(200)]
plt.plot(aa, inv_y[:200], marker='.', label="actual")
plt.plot(aa, inv_yhat[:200], 'r', label="prediction")
plt.ylabel('Global_active_power', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show()