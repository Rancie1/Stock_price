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



class EnsembleStockPredictor:
    """
    Ensemble model combining ARIMA (statistical) and other (deep learning) approaches
    """
    
    def __init__(self, arima_order=(5, 1, 0), ensemble_weights=None):
        """
        Initialize ensemble predictor
        
        Parameters:
        -----------
        arima_order : tuple
            ARIMA(p, d, q) order parameters
        ensemble_weights : dict
            Weights for combining predictions {'arima': 0.3, 'other': 0.7}
            If None, uses equal weights
        """
        self.arima_order = arima_order
        self.ensemble_weights = ensemble_weights or {'arima': 0.5, 'other': 0.5}
        self.arima_model = None
        self.other_model = None
        self.arima_fitted = None
        self.train_data_series = None  # Store training data for retraining
        
    def check_stationarity(self, timeseries, name='Series'):
        """
        Check if time series is stationary using Augmented Dickey-Fuller test
        """
        print(f"\nStationarity Test for {name}:")
        print("-" * 50)
        
        result = adfuller(timeseries.dropna())
        print(f'ADF Statistic: {result[0]:.4f}')
        print(f'p-value: {result[1]:.4f}')
        print(f'Critical Values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value:.3f}')
        
        if result[1] <= 0.05:
            print("✓ Series is stationary (reject null hypothesis)")
            return True
        else:
            print("✗ Series is non-stationary (fail to reject null hypothesis)")
            return False
    
    def find_best_arima_order(self, timeseries, max_p=5, max_d=2, max_q=5):
        """
        Find best ARIMA order using AIC criterion
        """
        print("\nSearching for best ARIMA parameters...")
        best_aic = np.inf
        best_order = None
        best_model = None
        
        # Test different combinations
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(timeseries, order=(p, d, q))
                        fitted = model.fit()
                        
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                            best_model = fitted
                            
                    except Exception as e:
                        continue
        
        if best_order:
            print(f"Best ARIMA order: {best_order} (AIC: {best_aic:.2f})")
            return best_order, best_model
        else:
            print("No valid ARIMA model found, using default (1,1,1)")
            return (1, 1, 1), None
    
    def train_arima(self, train_data, target_column='Close', auto_order=False):
        """
        Train ARIMA model on historical data
        
        CRITICAL FIX: Don't manually difference the data!
        ARIMA handles differencing internally with the 'd' parameter.
        """
        print("\n" + "="*60)
        print("TRAINING ARIMA MODEL")
        print("="*60)
        
        # Extract target series - KEEP ORIGINAL DATA
        y_train = train_data[target_column].dropna()
        self.train_data_series = y_train.copy()  # Store for later use
        
        print(f"Training data shape: {y_train.shape}")
        print(f"Training data range: {y_train.min():.2f} to {y_train.max():.2f}")
        
        # Check stationarity (for information only)
        is_stationary = self.check_stationarity(y_train, target_column)
        
        if not is_stationary:
            print("\n⚠ Series is non-stationary, but ARIMA will handle this with 'd' parameter")
        
        # Find best order if requested
        if auto_order:
            print("\nSearching for optimal ARIMA parameters (this may take a while)...")
            self.arima_order, best_fitted = self.find_best_arima_order(
                y_train, max_p=5, max_d=2, max_q=5
            )
            if best_fitted is not None:
                self.arima_fitted = best_fitted
                self.arima_model = best_fitted.model
                print("Using auto-selected model")
                return self.arima_fitted
        
        print(f"\nUsing ARIMA order: {self.arima_order}")
        
        # Fit ARIMA model on ORIGINAL data (not differenced!)
        try:
            self.arima_model = ARIMA(y_train, order=self.arima_order)
            self.arima_fitted = self.arima_model.fit()
            
            print(f"\nARIMA Model fitted successfully!")
            print(f"AIC: {self.arima_fitted.aic:.2f}")
            print(f"BIC: {self.arima_fitted.bic:.2f}")
            
            # Make a test prediction to verify model works
            test_pred = self.arima_fitted.forecast(steps=1)
            print(f"Test prediction: ${test_pred.iloc[0]:.2f}")
            
        except Exception as e:
            print(f"⚠ ARIMA fitting failed: {e}")
            print("Trying simpler model: ARIMA(1,1,1)")
            self.arima_order = (1, 1, 1)
            try:
                self.arima_model = ARIMA(y_train, order=self.arima_order)
                self.arima_fitted = self.arima_model.fit()
                print("✓ Fallback model fitted successfully")
            except Exception as e2:
                print(f"✗ Fallback also failed: {e2}")
                raise
        
        return self.arima_fitted
    
    def predict_arima_rolling(self, test_data, target_column='Close', use_append=True):
        """
        Use rolling forecast for ARIMA predictions with error handling
        
        This is the proper way to evaluate ARIMA on test data:
        - Start with training data
        - Predict one step ahead
        - Add actual value to history
        - Repeat for each test point
        
        Parameters:
        -----------
        use_append : bool
            If True, uses the efficient append() method (faster, more stable)
            If False, refits model at each step (slower, potentially unstable)
        """
        print("\nMaking rolling ARIMA forecasts on test data...")
        print(f"Method: {'Append (efficient)' if use_append else 'Refit (traditional)'}")
        
        predictions = []
        test_values = test_data[target_column].dropna()
        
        if use_append:
            # EFFICIENT METHOD: Use append() to update model
            current_fitted = self.arima_fitted
            
            for i, actual_value in enumerate(test_values):
                try:
                    # Forecast one step ahead
                    forecast = current_fitted.forecast(steps=1)
                    pred_value = forecast.iloc[0] if hasattr(forecast, 'iloc') else forecast[0]
                    predictions.append(pred_value)
                    
                    # Append the actual value to the model
                    current_fitted = current_fitted.append([actual_value], refit=False)
                    
                except Exception as e:
                    # Fallback: use last predicted value with small adjustment
                    if len(predictions) > 0:
                        pred_value = predictions[-1] * (1 + np.random.normal(0, 0.001))
                    else:
                        pred_value = actual_value
                    predictions.append(pred_value)
                    
                    if i < 5:  # Only show first few errors
                        print(f"  Warning at step {i+1}: Using fallback prediction")
                
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i+1}/{len(test_values)} predictions")
        
        else:
            # TRADITIONAL METHOD: Refit at each step (with error handling)
            history = list(self.train_data_series.values)
            last_fitted = self.arima_fitted
            failed_count = 0
            
            for i, actual_value in enumerate(test_values):
                try:
                    # Try to fit model on current history
                    model = ARIMA(history, order=self.arima_order)
                    fitted = model.fit(method_kwargs={'warn_convergence': False})
                    
                    # Forecast one step ahead
                    forecast = fitted.forecast(steps=1)
                    pred_value = forecast.iloc[0] if hasattr(forecast, 'iloc') else forecast[0]
                    
                    # Store successful model
                    last_fitted = fitted
                    
                except Exception as e:
                    # If fitting fails, use the last successful model or simple fallback
                    if last_fitted is not None:
                        try:
                            forecast = last_fitted.forecast(steps=1)
                            pred_value = forecast.iloc[0] if hasattr(forecast, 'iloc') else forecast[0]
                        except:
                            pred_value = history[-1] * (1 + np.random.normal(0, 0.001))
                    else:
                        pred_value = history[-1]
                    
                    failed_count += 1
                    if failed_count <= 5:
                        print(f"  Warning at step {i+1}: Using fallback prediction")
                
                predictions.append(pred_value)
                history.append(actual_value)
                
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i+1}/{len(test_values)} predictions")
            
            if failed_count > 0:
                print(f"⚠ {failed_count} predictions used fallback methods")
        
        print(f"✓ Completed {len(predictions)} rolling forecasts")
        return np.array(predictions)
    

 

    
