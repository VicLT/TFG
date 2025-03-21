#!/usr/bin/env python
# coding: utf-8

# ## PART 0 - IMPORT TRAINED MODEL

# from tensorflow.keras.models import load_model
# regressor_model = load_model('trained_model.keras')

# ## PART 1 - DATA PREPROCESSING

# ### Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
import math
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, LSTM, Input, Dropout
from sklearn.metrics import mean_squared_error

# ### Import the training set

ds_train = yf.download('^GSPC', start='2010-01-01', end='2025-01-01', auto_adjust = True)
df_train = pd.DataFrame(ds_train)

df_train

# ### Column selection

df_closing_prices = df_train[['Close']]

df_closing_prices

# ### Scaling data

scaler = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = scaler.fit_transform(df_closing_prices)

training_set_scaled, training_set_scaled.shape

# ### Training sequences

def create_dataset(data, time_step):
    X_total, y_total = [], []
    
    for i in range(time_step, len(data)):
        X_total.append(data[i-time_step:i, 0])  # Los últimos 60 días
        y_total.append(data[i, 0])  # El precio de cierre del día siguiente
    
    return np.array(X_total), np.array(y_total)

X_total, y_total = create_dataset(training_set_scaled, 60)

pd.DataFrame(X_total)

pd.DataFrame(y_total)

# ### Split into training set and testing set

X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.1, shuffle=False)

pd.DataFrame(X_train)

pd.DataFrame(y_train)

pd.DataFrame(X_test)

pd.DataFrame(y_test)

# ### Reshaping for 3D

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_train, X_train.shape

# ## PART 2 - BUILDING AND TRAINING THE RNN

# ### Initialising the RNN

regressor_model = Sequential()

# ### Adding the input layer

regressor_model.add(Input(shape = (X_train.shape[1], 1)))

# ### Adding the first hidden LSTM layer

regressor_model.add(LSTM(units = 50, return_sequences = True))
regressor_model.add(Dropout(0.2))

# ### Adding the last hidden LSTM layer

regressor_model.add(LSTM(units = 50))
regressor_model.add(Dropout(0.2))

# ### Adding the output layer

regressor_model.add(Dense(units = 1))

# ### Compiling the RNN

regressor_model.compile(optimizer = "adam", loss = "mean_squared_error", metrics=["mae", "mse"])

# ### Model summary

regressor_model.summary()

# ### Fitting the RNN to the training set

history = regressor_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

regressor_model.summary()

# ## PART 3 - MAKING THE PREDICTIONS AND VISUALISING THE RESULTS

# ### Calculating prediction and transform the results into 2D arrays

# Calcular las predicciones
predicted_stock_price = regressor_model.predict(X_test)

pd.DataFrame(predicted_stock_price)

# Invertir la normalización para obtener los precios en su escala original
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
real_stock_price = scaler.inverse_transform(y_test.reshape(-1, 1))

print(predicted_stock_price.shape, real_stock_price.shape)

# ### Preparing the results

dates = pd.date_range(start=df_closing_prices.index[-372], periods=372, freq='B')

dates

df_results = pd.DataFrame({
    'Date': dates,
    'Real': real_stock_price.flatten(),
    'Predicted': predicted_stock_price.flatten()
})

df_results

# ### Visualising the results

plt.figure(figsize=(12, 7))

plt.plot(df_results['Date'], df_results['Real'], color='red', label='Real')
plt.plot(df_results['Date'], df_results['Predicted'], color='blue', label='Predicted')

plt.title('S&P 500 Stock Price (2023-2024)')

plt.ylabel('Price')
plt.xticks(rotation = 45)

plt.legend()
plt.grid(True)

plt.show()

# ## PART 4 - EVALUATING THE RNN

# ### LOSS -> MSE (Mean Squared Error)

plt.figure(figsize=(12, 7))

plt.plot(history.history['mse'], color='blue', label='Training')
plt.plot(history.history['val_mse'], color='red', label='Validation')

plt.title('MSE (Mean Squared Error)')
plt.xlabel('Epochs')
plt.grid(True)
plt.legend()

plt.show()

# ### MAE (Mean Absolute Error)

plt.figure(figsize=(12, 7))

plt.plot(history.history['mae'], color='blue', label='Training')
plt.plot(history.history['val_mae'], color='red', label='Validation')

plt.title('MAE (Mean Absolute Error)')
plt.xlabel('Epochs')
plt.grid(True)
plt.legend()

plt.show()

test_loss, test_mae, test_mse = regressor_model.evaluate(X_test, y_test)

print("MAE:", test_mae)
print("MSE:", test_mse)

# ### RMSE (Root Mean Squared Error)

rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

print("RMSE:", rmse)

# ### Relative RMSE

price_range = max(real_stock_price) - min(real_stock_price)
relative_rmse = rmse / price_range

print("Relative RMSE:", relative_rmse)

# ## PART 5 - EXPORT TRAINED MODEL AND SCALER

regressor_model.save("trained_model.keras")

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# ## PART 6 - CALCULATING NEW HORIZON TIME

# ### Last 60-day sequence (escalation)

last_days = training_set_scaled[-61:].astype(np.float32)

pd.DataFrame(last_days)

X_test_january, y_test_january = create_dataset(last_days, 60)

pd.DataFrame(X_test_january)

pd.DataFrame(y_test_january)

# ### Recursive calculations January 2025

# Copy of the last 60 values of X_test
X_test_january = training_set_scaled[-60:].flatten()

# List for storing future predictions
january_2025_prediction = []

# Difference between the last actual value and the previous one (to adjust trend correctly)
real_trend = training_set_scaled[-1] - training_set_scaled[-2]

for _ in range(7):  # 20 financial days ~ 1 normal month
    # Convert sequence to TensorFlow tensor
    sequence_for_pred_tensor = tf.convert_to_tensor(X_test_january.reshape(1, 60, 1), dtype=tf.float32)

    # Predict the following value
    next_day_prediction = regressor_model(sequence_for_pred_tensor, training=False).numpy().flatten()[0]

    # Compare the last prediction with the last actual value
    last_real_value = training_set_scaled[-1]  # Last actual value available
    last_predicted_value = january_2025_prediction[-1] if january_2025_prediction else last_real_value

    # Trend adjustment with actual comparison
    trend_adjustment = (last_real_value - last_predicted_value) * 0.5  # Corrects the deviation at each iteration
    next_day_prediction += np.random.normal(0, 0.005) + trend_adjustment

    # Save the prediction
    january_2025_prediction.append(next_day_prediction)

    # Update the sequence for the next iteration
    X_test_january = np.append(X_test_january[1:], next_day_prediction)

# Transform predictions to original scale
january_2025_prediction_original_scale = scaler.inverse_transform(np.array(january_2025_prediction).reshape(-1, 1))

january_2025_prediction_original_scale.shape

# ### January 2025 actual

ds_january_2025_real = yf.download('^GSPC', start='2025-01-01', end='2025-01-14', auto_adjust = True)
df_january_2025_real = pd.DataFrame(ds_january_2025_real[['Close']])

df_january_2025_real.shape

# ### Combining actual and predicted data

df_january_2025_combined = df_january_2025_real.copy()

df_january_2025_combined['Predicted'] = january_2025_prediction_original_scale

df_january_2025_combined.index.name = 'Date'
df_january_2025_combined.columns = ['Real', 'Predicted']

df_january_2025_combined

# ### Actual vs. predictive chart

plt.figure(figsize=(12, 7))

plt.plot(df_january_2025_combined.index,
         df_january_2025_combined['Real'],
         color='red',
         label='Real')

plt.plot(df_january_2025_combined.index,
         df_january_2025_combined['Predicted'],
         color='blue',
         label='Predicted')

plt.title('S&P 500 (January 2025)')
plt.ylabel('Close')
plt.xticks(ticks=df_january_2025_combined.index,
           labels=df_january_2025_combined.index.strftime('%Y-%m-%d'),
           rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

# ### RMSE

new_rmse = math.sqrt(mean_squared_error(df_january_2025_real, january_2025_prediction_original_scale))

print("RMSE:", new_rmse)

# ### Relative RMSE

new_price_range = df_january_2025_real.max().iloc[0] - df_january_2025_real.min().iloc[0]
new_relative_rmse = new_rmse / new_price_range

print("Relative RMSE:", relative_rmse)
