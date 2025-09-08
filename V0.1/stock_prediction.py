# stock_prediction.py
# Stock prediction now using the create_model function

from load_function import load_and_process_stock_data
from model_builder import create_model
from tensorflow.keras.layers import LSTM, GRU

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
print("STOCK PREDICTION")
print("="*80)

# Load and Process Data
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

# Function to train and evaluate a model
def train_and_evaluate_model(model, model_name, epochs=20, batch_size=32):
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

# Model 1: Simple LSTM with custom layer names
print("\n" + "="*60)
print("BUILDING MODEL 1: SIMPLE LSTM")
print("="*60)

try:
    model1 = create_model(
        sequence_length=X_train.shape[1],
        n_features=X_train.shape[2],
        units=50,
        n_layers=2,
        dropout=0.2,
        layer_names=['encoder_lstm', 'decoder_lstm']
    )
    
    print(f"Model 1 Layer names: {[layer.name for layer in model1.layers]}")
    
    results1 = train_and_evaluate_model(model1, "Simple LSTM", epochs=20, batch_size=16)
    
    print("✅ Model 1 completed successfully!")
    
except Exception as e:
    print(f"❌ Error in Model 1: {e}")

# Model 2: Deep LSTM with auto-generated names
print("\n" + "="*60)
print("BUILDING MODEL 2: DEEP LSTM")
print("="*60)

try:
    model2 = create_model(
        sequence_length=X_train.shape[1],
        n_features=X_train.shape[2],
        units=64,
        n_layers=4,
        dropout=0.3
        # No layer_names parameter = auto-generated names
    )
    
    print(f"Model 2 Layer names: {[layer.name for layer in model2.layers]}")
    
    results2 = train_and_evaluate_model(model2, "Deep LSTM", epochs=20, batch_size=16)
    
    print("✅ Model 2 completed successfully!")
    
except Exception as e:
    print(f"❌ Error in Model 2: {e}")

# Model 3: Bidirectional LSTM
print("\n" + "="*60)
print("BUILDING MODEL 3: BIDIRECTIONAL LSTM")
print("="*60)

try:
    model3 = create_model(
        sequence_length=X_train.shape[1],
        n_features=X_train.shape[2],
        units=128,
        n_layers=2,
        dropout=0.25,
        bidirectional=True,
        layer_names=['bidirectional_encoder', 'bidirectional_processor']
    )
    
    print(f"Model 3 Layer names: {[layer.name for layer in model3.layers]}")
    
    results3 = train_and_evaluate_model(model3, "Bidirectional LSTM", epochs=20, batch_size=16)
    
    print("✅ Model 3 completed successfully!")
    
except Exception as e:
    print(f"❌ Error in Model 3: {e}")

# Model 4: GRU Model
print("\n" + "="*60)
print("BUILDING MODEL 4: GRU MODEL")
print("="*60)

try:
    model4 = create_model(
        sequence_length=X_train.shape[1],
        n_features=X_train.shape[2],
        units=100,
        cell=GRU,  # Using GRU instead of LSTM
        n_layers=3,
        dropout=0.2,
        layer_names=['gru_primary', 'gru_secondary', 'gru_final']
    )
    
    print(f"Model 4 Layer names: {[layer.name for layer in model4.layers]}")
    
    results4 = train_and_evaluate_model(model4, "GRU Model", epochs=20, batch_size=16)
    
    print("✅ Model 4 completed successfully!")
    
except Exception as e:
    print(f"❌ Error in Model 4: {e}")

# Compare models and plot the best one
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

models_results = []
if 'results1' in locals():
    models_results.append(("Simple LSTM", results1))
if 'results2' in locals():
    models_results.append(("Deep LSTM", results2))
if 'results3' in locals():
    models_results.append(("Bidirectional LSTM", results3))
if 'results4' in locals():
    models_results.append(("GRU Model", results4))

if models_results:
    print("\nModel Performance Comparison:")
    print("-" * 50)
    print(f"{'Model Name':<20} {'RMSE ($)':<12} {'MAE ($)':<12}")
    print("-" * 50)
    
    for name, result in models_results:
        print(f"{name:<20} {result['test_rmse']:<12.2f} {result['test_mae']:<12.2f}")
    
    # Find and plot best model
    best_model_name, best_result = min(models_results, key=lambda x: x[1]['test_rmse'])
    print(f"\nBest performing model: {best_model_name} (RMSE: ${best_result['test_rmse']:.2f})")
    
    # Plot the best model's results
    plt.figure(figsize=(14, 8))
    plt.plot(best_result['y_test_original'], color="black", label=f"Actual {COMPANY} Price", linewidth=2)
    plt.plot(best_result['test_predictions'], color="green", label=f"Predicted {COMPANY} Price", linewidth=2)
    plt.title(f"{COMPANY} Share Price Prediction - {best_model_name}")
    plt.xlabel("Test Days")
    plt.ylabel(f"{COMPANY} Share Price ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Future prediction with the best model
    print("\n" + "="*60)
    print("FUTURE PRICE PREDICTION")
    print("="*60)
    
    best_model = best_result['model']
    last_sequence = metadata['last_sequence']
    future_input = last_sequence[-PREDICTION_DAYS:].reshape(1, PREDICTION_DAYS, X_train.shape[2])
    
    target_scaler = metadata['scalers'][metadata['target_column']]
    future_prediction_scaled = best_model.predict(future_input)
    future_prediction = target_scaler.inverse_transform(future_prediction_scaled)
    
    print(f"Next day price prediction: ${future_prediction[0][0]:.2f}")
    
    # Compare with last known price
    last_actual = best_result['y_test_original'][-1][0]
    price_change = future_prediction[0][0] - last_actual
    price_change_percent = (price_change / last_actual) * 100
    
    print(f"Last known price: ${last_actual:.2f}")
    print(f"Predicted change: ${price_change:.2f} ({price_change_percent:+.2f}%)")

print("\n" + "="*80)
print("SIMPLE ENHANCED MODEL TESTING COMPLETE")
print("="*80)
print("✅ Enhanced create_model function with layer naming support")
print("✅ Multiple architectures tested with custom and auto-generated names")
print("✅ Simple and clean implementation")
print("="*80)
