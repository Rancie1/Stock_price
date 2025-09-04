# Example usage function

from load_function import load_and_process_stock_data

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