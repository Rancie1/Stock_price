import numpy as np

def predict_multistep_prices(model, last_sequence, n_steps, k_days, target_scaler, target_feature_idx=0):
    """
    Predict stock prices for k days into the future using multistep prediction.
    
    Parameters:
    -----------
    model : keras model
        Trained model for making predictions
    last_sequence : np.ndarray
        Last sequence of features with shape (n_steps, n_features)
    n_steps : int
        Number of time steps the model expects (sequence length)
    k_days : int
        Number of days to predict into the future
    target_scaler : sklearn.preprocessing.MinMaxScaler
        Scaler to inverse transform predictions back to original scale
    target_feature_idx : int
        Index of the target feature in the feature array (usually 0 for Close price)
        
    Returns:
    --------
    predictions : np.ndarray
        Array of predicted prices for k days, shape (k_days,)
    """
    
    if last_sequence.shape[0] < n_steps:
        raise ValueError(f"last_sequence must have at least {n_steps} time steps, got {last_sequence.shape[0]}")
    
    # Take the last n_steps from the sequence
    current_sequence = last_sequence[-n_steps:].copy()
    predictions = []
    


    
    for day in range(k_days):
        # Reshape for model input: (1, n_steps, n_features)
        model_input = current_sequence.reshape(1, n_steps, -1)
        
        # Make prediction for next day
        prediction_scaled = model.predict(model_input, verbose=0)
        
        # Convert back to original scale
        prediction = target_scaler.inverse_transform(prediction_scaled)[0][0]
        predictions.append(prediction)
        
        # Update sequence for next prediction
        # Create next time step by copying last features and updating target
        next_features = current_sequence[-1].copy()
        
        # Scale the prediction back to use in the sequence
        scaled_prediction = target_scaler.transform([[prediction]])[0][0]
        next_features[target_feature_idx] = scaled_prediction
        
        # Shift the sequence: remove first row, add new row
        current_sequence = np.vstack([current_sequence[1:], next_features])
        
        print(f"Day {day + 1}: ${prediction:.2f}")
    
    return np.array(predictions)




def predict_multivariate_closing_price(model, last_sequence, n_steps, future_features, 
                                     target_scaler, features_list, target_column='Close'):
    """
    Predict closing price using multivariate input features for a specific future day.
    
    This function uses all available features (Open, High, Low, Close, Adj Close, Volume)
    to predict the closing price for a specified day in the future.
    
    Parameters:
    -----------
    model : keras model
        Trained model for making predictions
    last_sequence : np.ndarray
        Historical sequence of features with shape (n_steps, n_features)
    n_steps : int
        Number of time steps the model expects
    future_features : dict or np.ndarray
        Either a dictionary with feature names as keys and predicted values,
        or a numpy array with shape (n_features,) representing the future day's features
    target_scaler : sklearn.preprocessing.MinMaxScaler
        Scaler for the target column (Close price)
    features_list : list
        List of feature names in the order they appear in the data
    target_column : str
        Name of the target column to predict
        
    Returns:
    --------
    prediction : float
        Predicted closing price for the future day
    confidence_info : dict
        Dictionary containing prediction details and confidence metrics
    """
    
    
    # Prepare the sequence for prediction
    current_sequence = last_sequence[-n_steps:].copy()
    
                
    if isinstance(future_features, (np.ndarray, list)):
        future_feature_array = np.array(future_features)
        print("Input features for prediction day:")
        for i, (feature, value) in enumerate(zip(features_list, future_feature_array)):
            print(f"  {feature}: {value:.4f}")
    else:
        raise ValueError("future_features must be numpy array or list")
    
    # Create new sequence by shifting and adding the future features
    new_sequence = np.vstack([current_sequence[1:], future_feature_array])
    
    # Reshape for model input
    model_input = new_sequence.reshape(1, n_steps, -1)
    
    # Make prediction
    prediction_scaled = model.predict(model_input, verbose=0)
    prediction = target_scaler.inverse_transform(prediction_scaled)[0][0]
    
   
    
    print(f"\n PREDICTION RESULTS:")
    print(f"Predicted {target_column} Price: ${prediction:.2f}")

    
    return prediction




def predict_multivariate_multistep_prices(model, last_sequence, n_steps, k_days, 
                                        scalers_dict, features_list, 
                                        target_column='Close', feature_evolution='auto'):
    """
    Combined multivariate and multistep prediction function.
    
    This function uses multiple features (multivariate) to predict multiple days ahead (multistep).
    It intelligently evolves all features over time, not just the target variable.
    
    Parameters:
    -----------
    model : keras model
        Trained model for making predictions
    last_sequence : np.ndarray
        Historical sequence of all features with shape (n_steps, n_features)
    n_steps : int
        Number of time steps the model expects
    k_days : int
        Number of days to predict into the future
    scalers_dict : dict
        Dictionary of scalers for each feature
    features_list : list
        List of feature names in order
    target_column : str
        Name of the target column to predict
    feature_evolution : str or dict
        How to evolve non-target features:
        - 'auto': automatically evolve based on historical patterns
        - 'static': keep features at last known values
        - dict: specify evolution rules for each feature
        
    Returns:
    --------
    predictions : dict
        Dictionary containing:
        - 'target_prices': array of predicted target prices
        - 'all_features': array of all evolved features for each day
        - 'feature_evolution': how each feature evolved
    """
    
    if last_sequence.shape[0] < n_steps:
        raise ValueError(f"last_sequence must have at least {n_steps} time steps, got {last_sequence.shape[0]}")
    
    # Initialize
    current_sequence = last_sequence[-n_steps:].copy()
    target_idx = features_list.index(target_column)
    target_scaler = scalers_dict[target_column]
    
    predictions = {
        'target_prices': [],
        'all_features': [],
        'feature_evolution': {feature: [] for feature in features_list}
    }
    
    print(f"Combined Multivariate-Multistep Prediction: {k_days} days ahead")
    print(f"Using features: {features_list}")
    print("-" * 60)
    
    for day in range(k_days):
        # Make prediction using current sequence
        model_input = current_sequence.reshape(1, n_steps, -1)
        prediction_scaled = model.predict(model_input, verbose=0)
        
        # Convert target prediction back to original scale
        target_price = target_scaler.inverse_transform(prediction_scaled)[0][0]
        predictions['target_prices'].append(target_price)
        
        # Evolve all features for the next time step
        next_features = current_sequence[-1].copy()
        
        # Update target feature with prediction
        target_scaled = target_scaler.transform([[target_price]])[0][0]
        next_features[target_idx] = target_scaled
        
        # Evolve other features based on strategy
        for i, feature in enumerate(features_list):
            if i == target_idx:
                continue  # Already updated target
                
            if feature_evolution == 'static':
                # Keep feature at last known value
                pass  # next_features[i] already has last value
                
            elif feature_evolution == 'auto':
                # Automatically evolve based on recent trends
                if current_sequence.shape[0] >= 3:
                    # Calculate trend from last 3 points
                    recent_values = current_sequence[-3:, i]
                    trend = np.mean(np.diff(recent_values))
                    next_features[i] = current_sequence[-1, i] + trend
                    
                    # Apply some constraints to prevent unrealistic values
                    if feature in ['Volume']:
                        # Volume shouldn't go negative
                        next_features[i] = max(next_features[i], 0.01)
                    elif feature in ['High', 'Low', 'Open']:
                        # Price features should stay within reasonable bounds relative to Close
                        target_unscaled = target_price
                        if scalers_dict[feature]:
                            # Convert to check bounds
                            feature_unscaled = scalers_dict[feature].inverse_transform([[next_features[i]]])[0][0]
                            
                            # High should be >= Close, Low should be <= Close
                            if feature == 'High':
                                feature_unscaled = max(feature_unscaled, target_unscaled)
                            elif feature == 'Low':
                                feature_unscaled = min(feature_unscaled, target_unscaled)
                            elif feature == 'Open':
                                # Open can vary but shouldn't be too extreme
                                feature_unscaled = np.clip(feature_unscaled, 
                                                         target_unscaled * 0.95, 
                                                         target_unscaled * 1.05)
                            
                            # Scale back
                            next_features[i] = scalers_dict[feature].transform([[feature_unscaled]])[0][0]
            
            elif isinstance(feature_evolution, dict) and feature in feature_evolution:
                # Custom evolution rule for this feature
                evolution_rule = feature_evolution[feature]
                if isinstance(evolution_rule, (int, float)):
                    # Fixed change
                    next_features[i] = current_sequence[-1, i] + evolution_rule
                elif callable(evolution_rule):
                    # Custom function
                    next_features[i] = evolution_rule(current_sequence[:, i], day)
        
        # Store evolved features (in original scale for interpretation)
        evolved_features_original = {}
        for i, feature in enumerate(features_list):
            if feature in scalers_dict:
                original_value = scalers_dict[feature].inverse_transform([[next_features[i]]])[0][0]
                evolved_features_original[feature] = original_value
                predictions['feature_evolution'][feature].append(original_value)
        
        predictions['all_features'].append(evolved_features_original)
        
        # Update sequence for next prediction
        current_sequence = np.vstack([current_sequence[1:], next_features])
        
        # Print daily results
        print(f"Day {day + 1}:")
        print(f"  {target_column}: ${target_price:.2f}")
        if day == 0:  # Show feature evolution for first day as example
            print(f"  Feature evolution sample:")
            for feat, val in list(evolved_features_original.items())[:3]:
                print(f"    {feat}: {val:.4f}")
    
    print(f"\nMultivariate-Multistep Prediction Complete!")
    print(f"Predicted {k_days} days with {len(features_list)} features")
    
    return predictions


# helper function to easily use the combined prediction
def run_combined_multivariate_multistep_analysis(model, metadata, prediction_days, k_days=5):
    """
    Easy-to-use wrapper for the combined prediction analysis
    """
    
    # Run the combined prediction
    combined_results = predict_multivariate_multistep_prices(
        model=model,
        last_sequence=metadata['last_sequence'],
        n_steps=prediction_days,
        k_days=k_days,
        scalers_dict=metadata['scalers'],
        features_list=metadata['features'],
        target_column=metadata['target_column'],
        feature_evolution='auto'
    )
    
    # Display summary
    target_prices = combined_results['target_prices']
    current_price = metadata['scalers'][metadata['target_column']].inverse_transform(
        [[metadata['last_sequence'][-1][0]]])[0][0]
    
    print(f"\nCombined Prediction Summary:")
    print("-" * 40)
    print(f"Current {metadata['target_column']}: ${current_price:.2f}")
    print(f"Final prediction (Day {k_days}): ${target_prices[-1]:.2f}")
    
    total_change = target_prices[-1] - current_price
    total_change_pct = (total_change / current_price) * 100
    print(f"Total change: ${total_change:+.2f} ({total_change_pct:+.2f}%)")
    
    # Show daily progression
    print(f"\nDaily progression:")
    for i, price in enumerate(target_prices):
        daily_change = price - current_price
        daily_change_pct = (daily_change / current_price) * 100
        print(f"Day {i+1}: ${price:.2f} ({daily_change_pct:+.2f}%)")
    
    return combined_results
