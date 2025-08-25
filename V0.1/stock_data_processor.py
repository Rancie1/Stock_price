import numpy as np
import pandas as pd
import yfinance as yf
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Optional, List

def load_and_process_stock_data(
    company_symbol: str,
    start_date: str,
    end_date: str,
    features: List[str] = None,
    handle_nan: str = 'drop',
    split_method: str = 'ratio',
    split_value: float = 0.8,
    split_date: str = None,
    random_split: bool = False,
    scale_features: bool = True,
    save_locally: bool = False,
    load_locally: bool = False,
    data_dir: str = 'stock_data',
    # Parameters for LSTM sequence creation
    n_steps: int = 60,
    lookup_step: int = 1,
    target_column: str = 'Close',
    create_sequences: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load and process stock data with multiple features and various processing options.
    
    Parameters:
    -----------
    company_symbol : str
        Stock symbol (e.g., 'CBA.AX', 'AAPL')
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str  
        End date in format 'YYYY-MM-DD'
    features : List[str], optional
        List of features to use ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        If None, uses all available features
    handle_nan : str, default 'drop'
        Method to handle NaN values: 'drop', 'fill_forward', 'fill_backward', 'fill_mean'
    split_method : str, default 'ratio'
        Method to split data: 'ratio', 'date', 'random'
    split_value : float, default 0.8
        Ratio for train/test split (0.8 means 80% train, 20% test)
    split_date : str, optional
        Date to split data when using 'date' method
    random_split : bool, default False
        Whether to randomly split data (used with 'random' split_method)
    scale_features : bool, default True
        Whether to scale features using MinMaxScaler
    save_locally : bool, default False
        Whether to save downloaded data locally
    load_locally : bool, default False
        Whether to try loading data from local storage first
    data_dir : str, default 'stock_data'
        Directory to save/load local data
    n_steps : int, default 60
        Number of time steps to look back for LSTM sequences (window size)
    lookup_step : int, default 1
        Number of days ahead to predict (1 = next day)
    target_column : str, default 'Close'
        Column name to use as prediction target
    create_sequences : bool, default True
        Whether to create LSTM sequences or return raw data
        
    Returns:
    --------
    Tuple containing:
    - X_train : Training sequences (n_samples, n_steps, n_features)
    - X_test : Test sequences  
    - y_train : Training targets (n_samples,)
    - y_test : Test targets
    - metadata : Dictionary containing scalers, last_sequence, and other info
    """
    
    # Create data directory if it doesn't exist
    if save_locally or load_locally:
        os.makedirs(data_dir, exist_ok=True)
    
    # Define filename for local storage
    filename = f"{company_symbol}_{start_date}_{end_date}.pkl"
    filepath = os.path.join(data_dir, filename)
    
    # Try to load data locally first if requested
    data = None
    if load_locally and os.path.exists(filepath):
        try:
            print(f"Loading data from local file: {filepath}")
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"Error loading local data: {e}")
            print("Proceeding to download data...")
    
    # Download data if not loaded locally
    if data is None:
        print(f"Downloading data for {company_symbol} from {start_date} to {end_date}")
        try:
            data = yf.download(company_symbol, start=start_date, end=end_date)
            
            # Handle MultiIndex columns (yfinance sometimes returns MultiIndex with ticker info)
            if isinstance(data.columns, pd.MultiIndex):
                # Flatten MultiIndex columns by taking the first level (feature names)
                data.columns = data.columns.get_level_values(0)
                print("Flattened MultiIndex columns")
            
            # Save data locally if requested
            if save_locally:
                print(f"Saving data to: {filepath}")
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
                    
        except Exception as e:
            raise Exception(f"Error downloading data: {e}")
    
    # Handle MultiIndex columns for locally loaded data as well
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        print("Flattened MultiIndex columns from local data")
    
    # Check if data is empty
    if data.empty:
        raise ValueError(f"No data found for {company_symbol} in the specified date range")
    
    print(f"Available columns: {data.columns.tolist()}")
    
    # Select features to use
    if features is None:
        # Use all available columns, but exclude 'Adj Close' if 'Close' exists to avoid confusion
        available_cols = data.columns.tolist()
        if 'Close' in available_cols and 'Adj Close' in available_cols:
            features = [col for col in available_cols if col != 'Adj Close']
        else:
            features = available_cols
        print(f"Using all available features: {features}")
    else:
        # Check if requested features exist in the data
        available_features = data.columns.tolist()
        missing_features = [f for f in features if f not in available_features]
        if missing_features:
            print(f"Available features: {available_features}")
            raise ValueError(f"Features not found in data: {missing_features}")
        print(f"Using specified features: {features}")
    
    # Extract the selected features
    processed_data = data[features].copy()
    
    print(f"Data shape before NaN handling: {processed_data.shape}")
    print(f"NaN values per column:\n{processed_data.isnull().sum()}")
    
    # Handle NaN values based on the specified method
    if handle_nan == 'drop':
        # Drop rows with any NaN values
        processed_data = processed_data.dropna()
        print(f"Data shape after dropping NaN: {processed_data.shape}")
        
    elif handle_nan == 'fill_forward':
        # Forward fill NaN values (use previous valid observation)
        processed_data = processed_data.fillna(method='ffill')
        
    elif handle_nan == 'fill_backward':
        # Backward fill NaN values (use next valid observation)  
        processed_data = processed_data.fillna(method='bfill')
        
    elif handle_nan == 'fill_mean':
        # Fill NaN values with column means
        processed_data = processed_data.fillna(processed_data.mean())
        
    else:
        raise ValueError("handle_nan must be one of: 'drop', 'fill_forward', 'fill_backward', 'fill_mean'")
    
    # Check for remaining NaN values
    remaining_nan = processed_data.isnull().sum().sum()
    if remaining_nan > 0:
        print(f"Warning: {remaining_nan} NaN values remain after processing")
    
    # Initialize metadata dictionary to store scalers and other info
    metadata = {
        'company_symbol': company_symbol,
        'start_date': start_date,
        'end_date': end_date,
        'features': features,
        'original_shape': processed_data.shape,
        'scalers': {},
        'n_steps': n_steps,
        'lookup_step': lookup_step,
        'target_column': target_column
    }
    
    # Scale features if requested
    if scale_features:
        print("Scaling features...")
        # Create a scaler for each feature
        scaled_data = processed_data.copy()
        
        for feature in features:
            scaler = MinMaxScaler(feature_range=(0, 1))
            # Reshape for scaler (needs 2D input)
            feature_data = processed_data[feature].values.reshape(-1, 1)
            scaled_feature = scaler.fit_transform(feature_data)
            scaled_data[feature] = scaled_feature.flatten()
            
            # Store scaler for this feature
            metadata['scalers'][feature] = scaler
            
        processed_data = scaled_data
        print("Feature scaling completed")
    
    # Ensure target column exists
    if target_column not in processed_data.columns:
        raise ValueError(f"Target column '{target_column}' not found in features: {features}")
    
    # Create sequences for LSTM if requested
    if create_sequences:
        print(f"Creating LSTM sequences with {n_steps} time steps, predicting {lookup_step} day(s) ahead...")
        
        # Add future target column by shifting the target column
        processed_data['future'] = processed_data[target_column].shift(-lookup_step)
        
        # Get the last n_steps + lookup_step rows for future prediction
        # This will be used to predict prices beyond our dataset
        last_sequence = processed_data[features].tail(n_steps + lookup_step).values
        metadata['last_sequence'] = last_sequence.astype(np.float32)
        
        # Remove NaN values created by the shift operation
        processed_data = processed_data.dropna()
        
        print(f"Data shape after adding future targets and removing NaN: {processed_data.shape}")
        
        # Create sequences using sliding window approach (similar to P1's deque method)
        from collections import deque
        sequences = deque(maxlen=n_steps)
        sequence_data = []
        
        # Convert to numpy for faster processing
        feature_data = processed_data[features].values
        target_data = processed_data['future'].values
        
        for i, (feature_row, target_value) in enumerate(zip(feature_data, target_data)):
            sequences.append(feature_row)
            
            # Once we have enough sequences, create a training sample
            if len(sequences) == n_steps:
                # Convert sequences to numpy array and append to sequence_data
                sequence_data.append([np.array(sequences), target_value])
        
        # Convert sequence_data to X and y arrays
        if len(sequence_data) == 0:
            raise ValueError(f"Not enough data to create sequences. Need at least {n_steps + lookup_step} samples.")
        
        X = np.array([seq[0] for seq in sequence_data])  # Shape: (n_samples, n_steps, n_features)
        y = np.array([seq[1] for seq in sequence_data])  # Shape: (n_samples,)
        
        print(f"Created {len(X)} sequences with shape: {X.shape}")
        
        # Store original data array for reference
        data_array = X, y
        
    else:
        # If not creating sequences, use simple array approach
        data_array = processed_data[features].values
        target_array = processed_data[target_column].values
        X = data_array[:-lookup_step]  # Remove last few samples that don't have future targets
        y = target_array[lookup_step:]  # Shift targets by lookup_step
        
        print(f"Created simple arrays - X shape: {X.shape}, y shape: {y.shape}")
    
    # Split data based on specified method
    if split_method == 'ratio':
        # Split by ratio (e.g., 80% train, 20% test)
        split_index = int(len(X) * split_value)
        
        if random_split:
            # Randomly shuffle indices before splitting
            indices = np.random.permutation(len(X))
            train_indices = indices[:split_index]
            test_indices = indices[split_index:]
            
            X_train = X[train_indices]
            X_test = X[test_indices]
            y_train = y[train_indices]
            y_test = y[test_indices]
        else:
            # Sequential split (first 80% train, last 20% test)
            X_train = X[:split_index]
            X_test = X[split_index:]
            y_train = y[:split_index]
            y_test = y[split_index:]
            
    elif split_method == 'date':
        # Split by specific date
        if split_date is None:
            raise ValueError("split_date must be provided when using 'date' split method")
            
        # For sequence data, we need to map back to original dates
        # This is more complex with sequences, so we'll use ratio-based approach with date logic
        # Convert split_date to datetime for comparison
        split_datetime = pd.to_datetime(split_date)
        
        # Find approximate split index based on date
        # This is approximate because sequences don't have direct date mapping
        original_dates = processed_data.index if create_sequences else processed_data.index[:-lookup_step]
        split_mask = original_dates < split_datetime
        split_index = split_mask.sum()
        
        # Ensure we don't exceed array bounds
        split_index = min(split_index, len(X) - 1)
        split_index = max(split_index, 1)
        
        X_train = X[:split_index]
        X_test = X[split_index:]
        y_train = y[:split_index]
        y_test = y[split_index:]
        
        if len(X_train) == 0:
            raise ValueError(f"No training data before split_date: {split_date}")
        if len(X_test) == 0:
            raise ValueError(f"No test data after split_date: {split_date}")
            
    else:
        raise ValueError("split_method must be one of: 'ratio', 'date'")
    
    print(f"Train data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Update metadata
    metadata.update({
        'split_method': split_method,
        'split_value': split_value,
        'split_date': split_date,
        'random_split': random_split,
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'create_sequences': create_sequences
    })
    
    return X_train, X_test, y_train, y_test, metadata

# Example usage function
def example_usage():
    """
    Example of how to use the load_and_process_stock_data function
    """
    try:
        # Example 1: LSTM sequences (like your original code)
        print("="*60)
        print("EXAMPLE 1: LSTM SEQUENCES (Compatible with your original model)")
        print("="*60)
        
        X_train, X_test, y_train, y_test, metadata = load_and_process_stock_data(
            company_symbol='CBA.AX',
            start_date='2020-01-01',
            end_date='2024-07-31',
            n_steps=60,  # Same as PREDICTION_DAYS in your original code
            lookup_step=1,  # Predict 1 day ahead
            target_column='Close',  # Predict Close price
            create_sequences=True,  # Create LSTM sequences
            save_locally=True
        )
        
        print(f"Features used: {metadata['features']}")
        print(f"Training sequences shape: {X_train.shape}")  # Should be (samples, 60, features)
        print(f"Training targets shape: {y_train.shape}")    # Should be (samples,)
        print(f"Test sequences shape: {X_test.shape}")
        print(f"Test targets shape: {y_test.shape}")
        print(f"Target column: {metadata['target_column']}")
        print(f"Last sequence for future prediction: {metadata['last_sequence'].shape}")
        
        # Example 2: Date-based split with sequences
        print("\n" + "="*60)
        print("EXAMPLE 2: DATE-BASED SPLIT WITH SEQUENCES")
        print("="*60)
        
        X_train2, X_test2, y_train2, y_test2, metadata2 = load_and_process_stock_data(
            company_symbol='CBA.AX',
            start_date='2020-01-01', 
            end_date='2024-07-31',
            features=['Open', 'High', 'Low', 'Close', 'Volume'],
            n_steps=50,  # Different window size
            split_method='date',
            split_date='2023-01-01',
            target_column='Close',
            load_locally=True  # Use previously saved data
        )
        
        print(f"Training sequences: {X_train2.shape}")
        print(f"Test sequences: {X_test2.shape}")
        print(f"Split date: {metadata2['split_date']}")
        
        # Example 3: Simple arrays (no sequences) for comparison
        print("\n" + "="*60)
        print("EXAMPLE 3: SIMPLE ARRAYS (NO SEQUENCES)")
        print("="*60)
        
        X_train3, X_test3, y_train3, y_test3, metadata3 = load_and_process_stock_data(
            company_symbol='CBA.AX',
            start_date='2020-01-01', 
            end_date='2024-07-31',
            create_sequences=False,  # Don't create sequences
            load_locally=True
        )
        
        print(f"Simple training data: {X_train3.shape}")
        print(f"Simple test data: {X_test3.shape}")
        
        # Example 4: Show how to use scalers for inverse transformation
        print("\n" + "="*60)
        print("EXAMPLE 4: INVERSE TRANSFORMATION USING SCALERS")
        print("="*60)
        
        if metadata['scalers']:
            # Get the scaler for the target column (Close)
            target_scaler = metadata['scalers'][metadata['target_column']]
            
            # Show original vs scaled values for first few predictions
            sample_predictions = y_test[:3].reshape(-1, 1)  # Take first 3 test targets
            original_values = target_scaler.inverse_transform(sample_predictions)
            
            print(f"Scaled target values: {sample_predictions.flatten()}")
            print(f"Original target values: ${original_values.flatten()}")
            
            # Show how you could use this in prediction
            print(f"\nThis is how you would convert model predictions back to real prices:")
            print(f"model_prediction = 0.65  # Example scaled prediction")
            print(f"real_price = target_scaler.inverse_transform([[0.65]])[0][0]")
            real_price = target_scaler.inverse_transform([[0.65]])[0][0]
            print(f"Real price: ${real_price:.2f}")
        
  
        
    except Exception as e:
        print(f"Error in example usage: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    example_usage()