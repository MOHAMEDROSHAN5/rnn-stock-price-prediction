# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

The task involves forecasting Google stock prices over time using RNN model for future stock price prediction. The dataset is a time series dataset consisting values of dates, opening stock rates, closing stock rates, High and low stock values. Based on the attributes given for the dataset, a stock price prediction model is to be developed.

## Design Steps

### Step 1:

1. Import necessary libraries

### Step 2:

2. Load the dataset and visualize the same.

### Step 3:

3. Perform pre-processing steps if required using MinMaxScaler.

### Step 4:

4. Create two lists for X_train and y_train and append a set of 60 readings in X_train, for which the 61st reading will be the first output in y_train.

### Step 5:

5. Create a model with a RNN layer and a dense layer

### Step 6:

6. Fit and compile the data for training set.

### Step 7:

7. Perform the same steps for Testing data and plot the graph for predictions of stock price.

## Name and Reg No: 

```
Name: MOHAMED ROSHAN S
Register Number: 212222040101

```

## Program


```py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential

dataset_train = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/DEEP LEARNING/ex5/trainset.csv')
dataset_train.columns
dataset_train.head()

train_set = dataset_train.iloc[:,1:2].values
type(train_set)
train_set.shape
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)
training_set_scaled.shape

X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))

X_train.shape
length = 60
n_features = 1

model = Sequential()
model.add(layers.SimpleRNN(50,input_shape=(length,n_features)))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mse')

print("Name: MOHAMED ROSHAN S    Register Number: 2122220101 ")
model.summary()
model.fit(X_train1,y_train,epochs=100, batch_size=32)
model.summary()
# Test Data
dataset_test = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/DEEP LEARNING/ex5/testset.csv')
test_set = dataset_test.iloc[:,1:2].values
test_set.shape

dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)

inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
y_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
  y_test.append(inputs_scaled[i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))

X_test.shape

predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)

print("Name: MOHAMED ROSHAN S    Register Number: 2122220101 ")
plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error as mse
mse = mse(y_test,predicted_stock_price)
print("Mean squared Error : ",mse)

```

## Output

### True Stock Price, Predicted Stock Price vs time

![alt](Screenshot%202024-03-31%20114055.png)

### Mean Square Error

![alt](Screenshot%202024-03-31%20114100.png)

## Result

Thus the stock price is predicted using Recurrent Neural Networks successfully.

