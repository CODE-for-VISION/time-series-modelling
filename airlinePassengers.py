 import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

def create_dataset(dataset, lookback):

	"""transform time series into prediction dataset"""
	
	"""
	args
		dataset : numpy array of time series
		lookback : size of window for prediction
	"""
	
	X, y = [], []
	
	for i in range(len(dataset)-lookback):
		feature = dataset[i:i+lookback]
		target  = dataset[i+1:i+lookback+1]
		X.append(feature)
		y.append(target)
	return torch.tensor(X), torch.tensor(y)
	
class AirModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
		self.linear = nn.Linear(50, 1)
	def forward(self, x):
		x, _ = self.lstm(x)
		x = self.linear(x)
		return x

df = pd.read_csv('airline-passengers.csv')
timeseries = df[["Passengers"]].values.astype('float32')

plt.plot(timeseries)
plt.show()

train_size = int(len(timeseries) * 0.67)
test_size = len(timeseries) - train_size

train, test = timeseries[:train_size], timeseries[train_size:]


 
lookback = 1
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test   = create_dataset(test, lookback=lookback)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


model = AirModel()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

n_epochs = 20000
for epoch in range(n_epochs):
	print(epoch)
	model.train()
	for X_batch, y_batch in loader:
		y_pred = model(X_batch)
		loss = loss_fn(y_pred, y_batch)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	# Validation
	if epoch % 10 != 0:
		continue
	model.eval()
	with torch.no_grad():
		y_pred = model(X_train)
		train_rmse = np.sqrt(loss_fn(y_pred, y_train))
		y_pred = model(X_test)
		test_rmse = np.sqrt(loss_fn(y_pred, y_test))
	print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
	
	
with torch.no_grad():
	
	train_plot = np.ones_like(timeseries) * np.nan
	y_pred = model(X_train)
	y_pred = y_pred[:, -1, :]
	train_plot[lookback:train_size] = model(X_train)[:, -1, :]
	
	test_plot = np.ones_like(timeseries) * np.nan
	test_plot[train_size+lookback:len(timeseries)] = model(X_test)[:, -1, :]
	
	
plt.plot(timeseries)
plt.plot(train_plot, c='r')
plt.plot(test_plot, c='g')
plt.show()

	
		
	


