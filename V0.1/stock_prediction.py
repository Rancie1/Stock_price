# File: fixed_enhanced_stock_prediction.py
# Fixed version that addresses the layer naming issue

from load_function import load_and_process_stock_data
from working_model_builder import build_deep_learning_model, create_lstm_model_config

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

# Configuration
COMPANY = 'CBA.AX'
TRAIN_START = '2020-01-01'
TRAIN_END = '2024-07-31'
PREDICTION_DAYS = 60

print("="*80)
print("FIXED ENHANCED STOCK PREDICTION WITH CUSTOMIZABLE DEEP LEARNING MODELS")
print("="*80)

# Load and Process Data (this part works fine)
print("\n" + "="*60)
print("LOADING AND PROCESSING DATA")
print("="*60)

X_train, X_test, y_train, y_test, metadata = load_and_process_stock_data(
    company_symbol=COMPANY,
    start_date=TRAIN_START,
    end_date=TRAIN_END,
    n_steps=PREDICTION_DAYS,
    target_column='Close',
    create_sequences=True,
    save_locally=True,
    load_locally=True,
    split_method='ratio',
    split_value=0.8
)

print(f"\nData Processing Complete:")
print(f"- Company: {metadata['company_symbol']}")
print(f"- Features: {metadata['features']}")
print(f"- Training sequences: {X_train.shape}")
print(f"- Test sequences: {X_test.shape}")
print(f"- Input shape for models: {(X_train.shape[1], X_train.shape[2])}")

# Function to train and evaluate a model
def train_and_evaluate_model(model, model_name, epochs=25, batch_size=32):
    """Train a model and return predictions with evaluation metrics"""
    
    print(f"\n" + "="*60)
    print(f"TRAINING {model_name.upper()}")
    print("="*60)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(X_test, y_test),
        shuffle=False
    )
    
    # Make predictions
    test_predictions_scaled = model.predict(X_test)
    
    # Convert back to original scale
    target_scaler = metadata['scalers'][metadata['target_column']]
    test_predictions = target_scaler.inverse_transform(test_predictions_scaled)
    y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    test_rmse = np.sqrt(np.mean((y_test_original - test_predictions) ** 2))
    test_mae = np.mean(np.abs(y_test_original - test_predictions))
    
    print(f"\n{model_name} Performance:")
    print(f"Test RMSE: ${test_rmse:.2f}")
    print(f"Test MAE: ${test_mae:.2f}")
    
    return {
        'model': model,
        'history': history,
        'test_predictions': test_predictions,
        'y_test_original': y_test_original,
        'test_rmse': test_rmse,
        'test_mae': test_mae
    }

















# Test Model 1: Simple LSTM (corrected configuration)
print("\n" + "="*60)
print("BUILDING MODEL 1: SIMPLE LSTM")
print("="*60)

# Corrected configuration - no layer_name in the config, we'll let the function handle naming
simple_lstm_config = [
    {
        'layer_type': 'lstm',
        'units': 50,
        'layer_name': 'lstm_1',
        'return_sequences': True,
        'dropout': 0.2
    },
    {
        'layer_type': 'lstm',
        'units': 50,
        'layer_name': 'lstm_2',
        'return_sequences': True,
        'dropout': 0.2
    },
    {
        'layer_type': 'lstm',
        'units': 50,
        'layer_name': 'lstm_3',
        'return_sequences': False,
        'dropout': 0.2
    }
]

try:
    model1 = build_deep_learning_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        layer_configs=simple_lstm_config,
        model_name="Simple_LSTM"
    )
    
    results1 = train_and_evaluate_model(model1, "Simple LSTM Model", epochs=10)  # Reduced epochs for testing
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(results1['y_test_original'], color="black", label=f"Actual {COMPANY} Price", linewidth=2)
    plt.plot(results1['test_predictions'], color="green", label=f"Predicted {COMPANY} Price", linewidth=2)
    plt.title(f"{COMPANY} Share Price Prediction - Simple LSTM Model")
    plt.xlabel("Test Days")
    plt.ylabel(f"{COMPANY} Share Price ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("✅ Model 1 completed successfully!")
    
except Exception as e:
    print(f"❌ Error in Model 1: {e}")
    import traceback
    traceback.print_exc()

# Test Model 2: Using helper function
print("\n" + "="*60)
print("BUILDING MODEL 2: LSTM WITH HELPER FUNCTION")
print("="*60)

try:
    # Use the helper function which should handle naming correctly
    lstm_config = create_lstm_model_config(
        lstm_layers=[64, 32],
        dense_layers=[25],
        dropout_rates=[0.2, 0.2]
    )
    
    model2 = build_deep_learning_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        layer_configs=lstm_config,
        model_name="Helper_LSTM"
    )
    
    results2 = train_and_evaluate_model(model2, "Helper LSTM Model", epochs=10)
    
    print("✅ Model 2 completed successfully!")
    
except Exception as e:
    print(f"❌ Error in Model 2: {e}")
    import traceback
    traceback.print_exc()

# Test Model 3: Minimal configuration
print("\n" + "="*60)
print("BUILDING MODEL 3: MINIMAL LSTM")
print("="*60)

try:
    # Most basic configuration to test the core functionality
    minimal_config = [
        {
            'layer_type': 'lstm',
            'units': 50,
            'layer_name': 'lstm_only'
        }
    ]
    
    model3 = build_deep_learning_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        layer_configs=minimal_config,
        model_name="Minimal_LSTM"
    )
    
    results3 = train_and_evaluate_model(model3, "Minimal LSTM Model", epochs=5)
    
    print("✅ Model 3 completed successfully!")
    
except Exception as e:
    print(f"❌ Error in Model 3: {e}")
    import traceback
    traceback.print_exc()

# Future prediction with the working model
if 'results1' in locals():
    print("\n" + "="*60)
    print("FUTURE PRICE PREDICTION")
    print("="*60)
    
    # Use model1 for future prediction
    last_sequence = metadata['last_sequence']
    future_input = last_sequence[-PREDICTION_DAYS:].reshape(1, PREDICTION_DAYS, X_train.shape[2])
    
    target_scaler = metadata['scalers'][metadata['target_column']]
    future_prediction_scaled = model1.predict(future_input)
    future_prediction = target_scaler.inverse_transform(future_prediction_scaled)
    
    print(f"Next day price prediction: ${future_prediction[0][0]:.2f}")
    
    # Compare with last known price
    last_actual = results1['y_test_original'][-1][0]
    price_change = future_prediction[0][0] - last_actual
    price_change_percent = (price_change / last_actual) * 100
    
    print(f"Last known price: ${last_actual:.2f}")
    print(f"Predicted change: ${price_change:.2f} ({price_change_percent:+.2f}%)")

print("\n" + "="*80)
print("TESTING COMPLETE")
print("="*80)
print("If no errors appeared above, the model builder is working correctly!")
print("You can now use it to build custom deep learning models for your stock predictions.")
print("="*80)
