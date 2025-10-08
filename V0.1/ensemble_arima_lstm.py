# ensemble_model.py
# Fixed ARIMA + LSTM Ensemble with proper ARIMA implementation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

from load_function import load_and_process_stock_data
from model_builder import create_model
from multi_prediction import predict_multistep_prices
from stock_prediction import train_and_evaluate_model
from tensorflow.keras.layers import LSTM
from ensemble import EnsembleStockPredictor as BaseEnsembleStockPredictor

class EnsembleStockPredictor(BaseEnsembleStockPredictor):
    """
    Ensemble model combining ARIMA (statistical) and other (deep learning) approaches
"""
    def __init__(self, arima_order=(5, 1, 0), ensemble_weights=None):
        """
        Initialize ensemble predictor for ARIMA + GRU
        """
        super().__init__(arima_order, ensemble_weights)
        # Override default weights to use 'gru' instead of 'lstm'
        if ensemble_weights is None:
            self.ensemble_weights = {'arima': 0.4, 'LSTM': 0.6}
        self.lstm_model = None  # Will store LSTM model

    
    def train_lstm(self, X_train, y_train, X_test, y_test, metadata,
                   units=64, n_layers=2, epochs=20, batch_size=16):
        """
        Train LSTM model using existing train_and_evaluate_model function
        """
        print("\n" + "="*60)
        print("TRAINING LSTM MODEL")
        print("="*60)
        
        self.lstm_model = create_model(
            sequence_length=X_train.shape[1],
            n_features=X_train.shape[2],
            units=units,
            cell=LSTM,
            n_layers=n_layers,
            dropout=0.2,
            layer_names=['lstm_encoder', 'lstm_decoder']
        )
        
        # Use existing train_and_evaluate_model function
        result = train_and_evaluate_model(
            self.lstm_model, "LSTM", epochs, batch_size, 
            X_train, y_train, X_test, y_test, metadata
        )
        
        return result['history']


 

    
    def ensemble_predict(self, arima_pred, lstm_pred):
        """
        Combine ARIMA and LSTM predictions using weighted average
        """
        w_arima = self.ensemble_weights['arima']
        w_lstm = self.ensemble_weights['lstm']
        
        ensemble_pred = w_arima * arima_pred + w_lstm * lstm_pred
        return ensemble_pred
    
    def evaluate_ensemble(self, y_true, arima_preds, lstm_preds, ensemble_preds):
        """
        Calculate metrics for all three approaches
        """
        def calc_metrics(y_true, y_pred):
            # Ensure arrays are same length
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            mae = np.mean(np.abs(y_true - y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
        
        results = {
            'ARIMA': calc_metrics(y_true, arima_preds),
            'LSTM': calc_metrics(y_true, lstm_preds),
            'Ensemble': calc_metrics(y_true, ensemble_preds)
        }
        
        return results
    
    def predict_next_day(self, last_lstm_sequence, n_steps, target_scaler, full_data):
        """
        Predict next day using ensemble approach
        Must retrain ARIMA with all available data for future prediction
        """
        # Retrain ARIMA on ALL data (train + test) for future prediction
        print("\nRetraining ARIMA on full dataset for future prediction...")
        temp_model = ARIMA(full_data['Close'].dropna(), order=self.arima_order)
        temp_fitted = temp_model.fit()
        
        # ARIMA prediction
        arima_forecast = temp_fitted.forecast(steps=1)
        arima_pred = arima_forecast.iloc[0] if hasattr(arima_forecast, 'iloc') else arima_forecast[0]
        
        # LSTM prediction
        lstm_input = last_lstm_sequence[-n_steps:].reshape(1, n_steps, -1)
        lstm_pred_scaled = self.lstm_model.predict(lstm_input, verbose=0)
        lstm_pred = target_scaler.inverse_transform(lstm_pred_scaled)[0][0]
        
        # Ensemble prediction
        ensemble_pred = self.ensemble_predict(arima_pred, lstm_pred)
        
        return {
            'arima': arima_pred,
            'lstm': lstm_pred,
            'ensemble': ensemble_pred
        }


# Main execution
if __name__ == "__main__":
    print("="*80)
    print(" ENSEMBLE MODEL: ARIMA + LSTM")
    print("="*80)
    
    # Configuration
    COMPANY = 'CBA.AX'
    TRAIN_START = '2020-01-01'
    TRAIN_END = '2024-07-31'
    PREDICTION_DAYS = 60
    
    # Load and process data
    print("\nLoading data...")
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
    
    # Load original data for ARIMA (needs time series, not sequences)
    print("\nLoading time series data for ARIMA...")
    import yfinance as yf
    data = yf.download(COMPANY, start=TRAIN_START, end=TRAIN_END)
    
    # Handle MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Split data for ARIMA
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    print(f"\nData split info:")
    print(f"Total data points: {len(data)}")
    print(f"Training points: {len(train_data)}")
    print(f"Test points: {len(test_data)}")
    
    # Initialize ensemble predictor
    ensemble = EnsembleStockPredictor(
        arima_order=(5, 1, 0),  # Try p=5, d=1, q=0 (AR model with differencing)
        ensemble_weights={'arima': 0.4, 'lstm': 0.6}  # Favor LSTM slightly
    )
    
    # Train ARIMA - Try auto_order=True for best results
    ensemble.train_arima(train_data, target_column='Close', auto_order=True)
    
    # Train LSTM
    ensemble.train_lstm(X_train, y_train, X_test, y_test, metadata, epochs=20)

    
    # Make predictions on test set using ROLLING FORECAST
    print("\n" + "="*60)
    print("ENSEMBLE PREDICTIONS ON TEST SET (ROLLING FORECAST)")
    print("="*60)
    
    target_scaler = metadata['scalers']['Close']
    
    # ARIMA predictions using proper rolling forecast with efficient append method
    arima_test_preds = ensemble.predict_arima_rolling(test_data, 'Close', use_append=True)
    
    # LSTM predictions
    lstm_test_preds_scaled = ensemble.lstm_model.predict(X_test, verbose=0)
    lstm_test_preds = target_scaler.inverse_transform(lstm_test_preds_scaled).flatten()
    
    # Get actual test values
    y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Align predictions (make sure all have same length)
    min_len = min(len(arima_test_preds), len(lstm_test_preds), len(y_test_actual))
    arima_test_preds = arima_test_preds[:min_len]
    lstm_test_preds = lstm_test_preds[:min_len]
    y_test_actual = y_test_actual[:min_len]
    
    print(f"\nAligned predictions length: {min_len}")
    
    # Ensemble predictions
    ensemble_test_preds = np.array([
        ensemble.ensemble_predict(a, l) 
        for a, l in zip(arima_test_preds, lstm_test_preds)
    ])
    
    # Evaluate
    metrics = ensemble.evaluate_ensemble(
        y_test_actual, arima_test_preds, lstm_test_preds, ensemble_test_preds
    )
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*60)
    print(f"{'Model':<15} {'RMSE ($)':<12} {'MAE ($)':<12} {'MAPE (%)':<12}")
    print("-" * 60)
    for model_name, model_metrics in metrics.items():
        print(f"{model_name:<15} {model_metrics['RMSE']:<12.2f} "
              f"{model_metrics['MAE']:<12.2f} {model_metrics['MAPE']:<12.2f}")
    
    # Determine which model performs best
    best_model = min(metrics.items(), key=lambda x: x[1]['RMSE'])
    print(f"\n✓ Best model: {best_model[0]} (RMSE: ${best_model[1]['RMSE']:.2f})")
    
    # Visualize predictions
    plt.figure(figsize=(15, 8))
    plt.plot(y_test_actual, 'k-', label='Actual Price', linewidth=2, alpha=0.8)
    plt.plot(arima_test_preds, 'b--', label='ARIMA', alpha=0.7, linewidth=1.5)
    plt.plot(lstm_test_preds, 'r--', label='LSTM', alpha=0.7, linewidth=1.5)
    plt.plot(ensemble_test_preds, 'g-', label='Ensemble', linewidth=2)
    plt.title(f'{COMPANY} Stock Price Prediction - Fixed Ensemble Model')
    plt.xlabel('Test Days')
    plt.ylabel('Price ($)')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Future single-day prediction
    print("\n" + "="*60)
    print("NEXT DAY PREDICTION")
    print("="*60)
    
    next_day_preds = ensemble.predict_next_day(
        last_lstm_sequence=metadata['last_sequence'],
        n_steps=PREDICTION_DAYS,
        target_scaler=target_scaler,
        full_data=data  # Pass full data for retraining
    )
    
    # Summary
    last_price = data['Close'].iloc[-1]
    print(f"\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"Current price: ${last_price:.2f}")
    print(f"\nNext Day Forecast:")
    print(f"  ARIMA:    ${next_day_preds['arima']:.2f}")
    print(f"  LSTM:     ${next_day_preds['lstm']:.2f}")
    print(f"  Ensemble: ${next_day_preds['ensemble']:.2f}")
    
    # Calculate changes
    arima_change = next_day_preds['arima'] - last_price
    lstm_change = next_day_preds['lstm'] - last_price
    ensemble_change = next_day_preds['ensemble'] - last_price
    
    print(f"\nPredicted Changes:")
    print(f"  ARIMA:    {arima_change:+.2f} ({(arima_change/last_price)*100:+.2f}%)")
    print(f"  LSTM:     {lstm_change:+.2f} ({(lstm_change/last_price)*100:+.2f}%)")
    print(f"  Ensemble: {ensemble_change:+.2f} ({(ensemble_change/last_price)*100:+.2f}%)")
    
    print("\n" + "="*80)
    print("ENSEMBLE MODELING COMPLETE")
    print("="*80)
