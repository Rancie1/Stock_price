# File: usage_examples.py
# Simple examples showing how to use the model builder correctly

from working_model_builder import build_deep_learning_model, create_lstm_model_config, create_cnn_lstm_model_config

# Example 1: Basic LSTM Model (like your original stock prediction)
def create_basic_lstm():
    """Create a basic LSTM model similar to the original stock_prediction.py"""
    
    config = [
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
    
    model = build_deep_learning_model(
        input_shape=(60, 5),  # 60 time steps, 5 features
        layer_configs=config,
        model_name="Basic_LSTM_Stock_Model"
    )
    
    return model

# Example 2: Deep LSTM with varying layer sizes
def create_deep_lstm():
    """Create a deeper LSTM model with decreasing layer sizes"""
    
    config = [
        {
            'layer_type': 'lstm',
            'units': 128,
            'layer_name': 'lstm_large',
            'dropout': 0.3
        },
        {
            'layer_type': 'lstm', 
            'units': 64,
            'layer_name': 'lstm_medium',
            'dropout': 0.2
        },
        {
            'layer_type': 'lstm',
            'units': 32,
            'layer_name': 'lstm_small',
            'dropout': 0.2
        },
        {
            'layer_type': 'dense',
            'units': 25,
            'layer_name': 'dense_output_prep',
            'activation': 'relu'
        }
    ]
    
    model = build_deep_learning_model(
        input_shape=(60, 5),
        layer_configs=config,
        learning_rate=0.001,
        model_name="Deep_LSTM_Stock_Model"
    )
    
    return model

# Example 3: Using Helper Functions
def create_lstm_with_helper():
    """Use the helper function to create LSTM models easily"""
    
    # This automatically creates the proper configuration
    config = create_lstm_model_config(
        lstm_layers=[64, 32, 16],     # Three LSTM layers with these sizes
        dense_layers=[25],            # One dense layer before output
        dropout_rates=[0.3, 0.2, 0.1]  # Different dropout for each LSTM layer
    )
    
    model = build_deep_learning_model(
        input_shape=(60, 5),
        layer_configs=config,
        model_name="Helper_LSTM_Model"
    )
    
    return model

# Example 4: Bidirectional LSTM
def create_bidirectional_lstm():
    """Create a bidirectional LSTM model"""
    
    config = create_lstm_model_config(
        lstm_layers=[64, 32],
        dense_layers=[50, 25], 
        use_bidirectional=True  # This makes all LSTM layers bidirectional
    )
    
    model = build_deep_learning_model(
        input_shape=(60, 5),
        layer_configs=config,
        model_name="Bidirectional_LSTM_Model"
    )
    
    return model

# Example 5: CNN-LSTM Hybrid
def create_cnn_lstm_hybrid():
    """Create a CNN-LSTM hybrid model"""
    
    config = create_cnn_lstm_model_config(
        conv_layers=[
            {'filters': 64, 'kernel_size': 3},
            {'filters': 32, 'kernel_size': 3}
        ],
        lstm_layers=[50, 25],
        dense_layers=[25]
    )
    
    model = build_deep_learning_model(
        input_shape=(60, 5),
        layer_configs=config,
        model_name="CNN_LSTM_Hybrid"
    )
    
    return model

# Example 6: Custom Complex Model
def create_custom_model():
    """Create a custom model with multiple layer types"""
    
    config = [
        # Feature extraction with Conv1D
        {
            'layer_type': 'conv1d',
            'units': 32,
            'kernel_size': 3,
            'layer_name': 'conv_features',
            'activation': 'relu'
        },
        # Dropout after convolution
        {
            'layer_type': 'dropout',
            'rate': 0.2,
            'layer_name': 'conv_dropout'
        },
        # Bidirectional GRU instead of LSTM
        {
            'layer_type': 'bidirectional',
            'units': 64,
            'rnn_type': 'gru',
            'layer_name': 'bi_gru'
        },
        # Final LSTM layer
        {
            'layer_type': 'lstm',
            'units': 32,
            'layer_name': 'lstm_final'
        },
        # Dense layers with batch normalization
        {
            'layer_type': 'dense',
            'units': 64,
            'layer_name': 'dense_1',
            'activation': 'relu'
        },
        {
            'layer_type': 'batchnormalization',
            'layer_name': 'batch_norm'
        },
        {
            'layer_type': 'dropout',
            'rate': 0.3,
            'layer_name': 'final_dropout'
        }
    ]
    
    model = build_deep_learning_model(
        input_shape=(60, 5),
        layer_configs=config,
        optimizer='adam',
        learning_rate=0.0005,
        model_name="Custom_Complex_Model"
    )
    
    return model

# Example 7: Simple GRU Model
def create_gru_model():
    """Create a GRU-based model instead of LSTM"""
    
    config = [
        {
            'layer_type': 'gru',
            'units': 50,
            'layer_name': 'gru_1',
            'return_sequences': True,
            'dropout': 0.2
        },
        {
            'layer_type': 'gru',
            'units': 30,
            'layer_name': 'gru_2',
            'dropout': 0.2
        },
        {
            'layer_type': 'dense',
            'units': 25,
            'layer_name': 'dense_final',
            'activation': 'relu'
        }
    ]
    
    model = build_deep_learning_model(
        input_shape=(60, 5),
        layer_configs=config,
        model_name="GRU_Stock_Model"
    )
    
    return model

# Testing function
def test_all_models():
    """Test all model creation functions"""
    
    models = {}
    
    try:
        print("Creating Basic LSTM...")
        models['basic_lstm'] = create_basic_lstm()
        print("✅ Basic LSTM created successfully")
    except Exception as e:
        print(f"❌ Basic LSTM failed: {e}")
    
    try:
        print("\nCreating Deep LSTM...")
        models['deep_lstm'] = create_deep_lstm()
        print("✅ Deep LSTM created successfully")
    except Exception as e:
        print(f"❌ Deep LSTM failed: {e}")
    
    try:
        print("\nCreating LSTM with Helper...")
        models['helper_lstm'] = create_lstm_with_helper()
        print("✅ Helper LSTM created successfully")
    except Exception as e:
        print(f"❌ Helper LSTM failed: {e}")
    
    try:
        print("\nCreating Bidirectional LSTM...")
        models['bi_lstm'] = create_bidirectional_lstm()
        print("✅ Bidirectional LSTM created successfully")
    except Exception as e:
        print(f"❌ Bidirectional LSTM failed: {e}")
    
    try:
        print("\nCreating CNN-LSTM Hybrid...")
        models['cnn_lstm'] = create_cnn_lstm_hybrid()
        print("✅ CNN-LSTM Hybrid created successfully")
    except Exception as e:
        print(f"❌ CNN-LSTM Hybrid failed: {e}")
    
    try:
        print("\nCreating Custom Model...")
        models['custom'] = create_custom_model()
        print("✅ Custom Model created successfully")
    except Exception as e:
        print(f"❌ Custom Model failed: {e}")
    
    try:
        print("\nCreating GRU Model...")
        models['gru'] = create_gru_model()
        print("✅ GRU Model created successfully")
    except Exception as e:
        print(f"❌ GRU Model failed: {e}")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(models)}/7 models created successfully")
    print(f"{'='*60}")
    
    return models


# Simple test function
def test_model_builder():
    """Test the model builder with a simple configuration"""
    
    print("Testing Model Builder...")
    
    # Test 1: Simple LSTM
    simple_config = [
        {
            'layer_type': 'lstm',
            'units': 50,
            'layer_name': 'test_lstm',
            'return_sequences': True,
            'dropout': 0.2
        },
        {
            'layer_type': 'lstm',
            'units': 30,
            'layer_name': 'test_lstm_2',
            'dropout': 0.2
        }
    ]
    
    try:
        model1 = build_deep_learning_model(
            input_shape=(60, 5),
            layer_configs=simple_config,
            model_name="Test_LSTM"
        )
        print("✅ Test 1 passed: Simple LSTM model created successfully")
        
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        return False
    
    # Test 2: Using helper function
    try:
        helper_config = create_lstm_model_config(
            lstm_layers=[64, 32],
            dense_layers=[25]
        )
        
        model2 = build_deep_learning_model(
            input_shape=(60, 5),
            layer_configs=helper_config,
            model_name="Test_Helper"
        )
        print("✅ Test 2 passed: Helper function works correctly")
        
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        return False
    
    # Test 3: Minimal config
    try:
        minimal_config = [
            {
                'layer_type': 'lstm',
                'units': 50,
                'layer_name': 'minimal_lstm'
            }
        ]
        
        model3 = build_deep_learning_model(
            input_shape=(60, 5),
            layer_configs=minimal_config,
            model_name="Test_Minimal"
        )
        print("✅ Test 3 passed: Minimal configuration works")
        
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Model builder is working correctly.")
    return True



    
if __name__ == "__main__":

    test_model_builder()

    
    # Test all models
    created_models = test_all_models()
    


    # Show a sample model summary
    if 'basic_lstm' in created_models:
        print("\nSample Model Summary (Basic LSTM):")
        print("-" * 40)
        created_models['basic_lstm'].summary()
