# File: stock_prediction.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)
# Enhanced: 25/08/2025 (v4) - Integrated with enhanced data processing

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following (best in a virtual env):
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance

from stock_data_processor import load_and_process_stock_data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


# Load and Process Data using load_and_process_stock_data

COMPANY = 'CBA.AX'
TRAIN_START = '2020-01-01'     # Start date to read
TRAIN_END = '2024-07-31'       # End date to read (extended for more data)
PREDICTION_DAYS = 60          # Number of days to look back

print("="*60)
print("LOADING AND PROCESSING DATA")
print("="*60)


# This replaces all the manual data loading, scaling, and sequence creation
X_train, X_test, y_train, y_test, metadata = load_and_process_stock_data(
    company_symbol=COMPANY,
    start_date=TRAIN_START,
    end_date=TRAIN_END,
    n_steps=PREDICTION_DAYS,    # Same as original PREDICTION_DAYS
    target_column='Close',      # Predict Close price
    create_sequences=True,      # Create LSTM sequences
    save_locally=True,          # Save for future use
    load_locally=True,         
    split_method='ratio',       # Use ratio split for consistency
    split_value=0.8            # 80% train, 20% test
)

print(f"\nData Processing Complete:")
print(f"- Company: {metadata['company_symbol']}")
print(f"- Features: {metadata['features']}")
print(f"- Training sequences: {X_train.shape}")
print(f"- Test sequences: {X_test.shape}")
print(f"- Target column: {metadata['target_column']}")

# load and process function has already:
# 1. Downloaded and cached the data locally
# 2. Handled NaN values
# 3. Scaled all features (0-1 range)
# 4. Created LSTM sequences with the correct shape
# 5. Split into train/test sets



# Build the Model

print("\n" + "="*60)
print("BUILDING LSTM MODEL")
print("="*60)

model = Sequential() # Basic neural network

# First LSTM layer with input shape matching our processed data
# X_train.shape is (samples, time_steps, features)
# We need input_shape=(time_steps, features)
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
# Original code used input_shape=(x_train.shape[1], 1) for single feature
# Load and Process function uses input_shape=(60, 5) for multiple features

model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1)) 
# Prediction of the next closing value of the stock price

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

print("Model architecture:")
model.summary()


# Train the Model

print("\n" + "="*60)
print("TRAINING MODEL")
print("="*60)

# Train the model with our processed data
# X_train and y_train are already in the correct format
history = model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=1,
                   validation_data=(X_test, y_test))


# Make Predictions on Test Data

print("\n" + "="*60)
print("MAKING PREDICTIONS")
print("="*60)

# Make predictions on test data
predicted_prices_scaled = model.predict(X_test)

# Convert predictions back to original price scale
# Get the scaler for the target column (Close)
target_scaler = metadata['scalers'][metadata['target_column']]
predicted_prices = target_scaler.inverse_transform(predicted_prices_scaled)

# Convert actual test targets back to original scale
y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1))

print(f"Predictions made for {len(predicted_prices)} test samples")
print(f"Sample predictions (first 5):")
for i in range(min(5, len(predicted_prices))):
    print(f"  Actual: ${y_test_original[i][0]:.2f}, Predicted: ${predicted_prices[i][0]:.2f}")


# Plot the Results

print("\n" + "="*60)
print("PLOTTING RESULTS")
print("="*60)

plt.figure(figsize=(12, 6))
plt.plot(y_test_original, color="black", label=f"Actual {COMPANY} Price", linewidth=2)
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price", linewidth=2)
plt.title(f"{COMPANY} Share Price Prediction")
plt.xlabel("Test Days")
plt.ylabel(f"{COMPANY} Share Price ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()




# Predict Future Prices

print("\n" + "="*60)
print("FUTURE PRICE PREDICTION")
print("="*60)

# Use the last sequence to predict the next day's price
# Load and process function stored this in metadata['last_sequence']
last_sequence = metadata['last_sequence']

# Take the last n_steps rows for prediction
future_input = last_sequence[-PREDICTION_DAYS:].reshape(1, PREDICTION_DAYS, X_train.shape[2])

# Make prediction
future_prediction_scaled = model.predict(future_input)
future_prediction = target_scaler.inverse_transform(future_prediction_scaled)

print(f"Next day price prediction: ${future_prediction[0][0]:.2f}")




