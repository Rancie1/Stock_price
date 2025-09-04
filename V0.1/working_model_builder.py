# File: working_model_builder.py
# Fixed Deep Learning Model Builder that definitely works

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, LSTM, GRU, SimpleRNN, 
    Conv1D, MaxPooling1D, GlobalMaxPooling1D,
    BatchNormalization, Bidirectional
)
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from typing import List, Dict, Optional, Tuple
import numpy as np

def build_deep_learning_model(
    input_shape: Tuple[int, int],
    layer_configs: List[Dict],
    output_units: int = 1,
    output_activation: str = None,
    optimizer: str = 'adam',
    learning_rate: float = 0.001,
    loss: str = 'mean_squared_error',
    metrics: List[str] = None,
    model_name: str = "CustomModel"
) -> tf.keras.Model:
    """
    Build a customizable deep learning model for time series prediction.
    
    Parameters:
    -----------
    input_shape : Tuple[int, int]
        Shape of input data (time_steps, features). For example: (60, 5)
        
    layer_configs : List[Dict]
        List of dictionaries, each defining a layer configuration.
        Each dictionary should contain:
        - 'layer_type': str - Type of layer ('lstm', 'gru', 'dense', 'conv1d', etc.)
        - 'units': int - Number of units/neurons in the layer
        - 'layer_name': str - Custom name for the layer
        - Additional layer-specific parameters (optional)
        
    Returns:
    --------
    tf.keras.Model
        Compiled Keras model ready for training
    """
    
    if not layer_configs:
        raise ValueError("layer_configs cannot be empty")
    
    if metrics is None:
        metrics = ['mae']
    
    print(f"Building {model_name} with input shape: {input_shape}")
    print(f"Number of layers: {len(layer_configs)}")
    
    # Initialize Sequential model
    model = Sequential(name=model_name)
    
    # Process each layer configuration
    for i, config in enumerate(layer_configs):
        layer_type = config.get('layer_type', '').lower()
        units = config.get('units')
        layer_name = config.get('layer_name', f"{layer_type}_{i+1}")
        
        if units is None:
            raise ValueError(f"'units' must be specified for layer {i+1}")
        
        print(f"Adding layer {i+1}: {layer_type} ({units} units) - {layer_name}")
        
        # Determine if next layer needs sequences
        next_needs_sequences = False
        if i < len(layer_configs) - 1:
            next_layer_type = layer_configs[i+1].get('layer_type', '').lower()
            next_needs_sequences = next_layer_type in ['lstm', 'gru', 'simplernn', 'conv1d', 'bidirectional']
        
        # Prepare layer arguments (excluding our custom keys)
        layer_kwargs = {}
        for key, value in config.items():
            if key not in ['layer_type', 'units', 'layer_name']:
                layer_kwargs[key] = value
        
        # Add input_shape to first layer
        if i == 0:
            layer_kwargs['input_shape'] = input_shape
        
        # Build different layer types
        if layer_type == 'lstm':
            if 'return_sequences' not in layer_kwargs:
                layer_kwargs['return_sequences'] = next_needs_sequences
            model.add(LSTM(units, name=layer_name, **layer_kwargs))
            
        elif layer_type == 'gru':
            if 'return_sequences' not in layer_kwargs:
                layer_kwargs['return_sequences'] = next_needs_sequences
            model.add(GRU(units, name=layer_name, **layer_kwargs))
            
        elif layer_type == 'simplernn' or layer_type == 'rnn':
            if 'return_sequences' not in layer_kwargs:
                layer_kwargs['return_sequences'] = next_needs_sequences
            model.add(SimpleRNN(units, name=layer_name, **layer_kwargs))
            
        elif layer_type == 'bidirectional':
            rnn_type = layer_kwargs.pop('rnn_type', 'lstm')
            if 'return_sequences' not in layer_kwargs:
                layer_kwargs['return_sequences'] = next_needs_sequences
                
            if rnn_type.lower() == 'lstm':
                inner_layer = LSTM(units, **layer_kwargs)
            elif rnn_type.lower() == 'gru':
                inner_layer = GRU(units, **layer_kwargs)
            else:
                inner_layer = SimpleRNN(units, **layer_kwargs)
                
            model.add(Bidirectional(inner_layer, name=layer_name))
            
        elif layer_type == 'conv1d':
            kernel_size = layer_kwargs.pop('kernel_size', 3)
            activation = layer_kwargs.pop('activation', 'relu')
            model.add(Conv1D(units, kernel_size, activation=activation, name=layer_name, **layer_kwargs))
            
        elif layer_type == 'maxpooling1d':
            pool_size = layer_kwargs.pop('pool_size', 2)
            model.add(MaxPooling1D(pool_size, name=layer_name))
            
        elif layer_type == 'globalmaxpooling1d':
            model.add(GlobalMaxPooling1D(name=layer_name))
            
        elif layer_type == 'dense':
            activation = layer_kwargs.pop('activation', 'relu')
            model.add(Dense(units, activation=activation, name=layer_name, **layer_kwargs))
            
        elif layer_type == 'dropout':
            rate = layer_kwargs.pop('rate', layer_kwargs.pop('dropout', 0.2))
            model.add(Dropout(rate, name=layer_name))
            
        elif layer_type == 'batchnormalization' or layer_type == 'batchnorm':
            model.add(BatchNormalization(name=layer_name))
            
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")
        
        # Add dropout if specified in config (for RNN layers)
        dropout_rate = config.get('dropout')
        if dropout_rate and layer_type in ['lstm', 'gru', 'simplernn', 'dense'] and layer_type != 'dropout':
            model.add(Dropout(dropout_rate, name=f"dropout_after_{layer_name}"))
    
    # Add output layer
    if output_activation:
        model.add(Dense(output_units, activation=output_activation, name='output_layer'))
    else:
        model.add(Dense(output_units, name='output_layer'))
    
    # Set up optimizer
    if isinstance(optimizer, str):
        if optimizer.lower() == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'rmsprop':
            opt = RMSprop(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = SGD(learning_rate=learning_rate)
        else:
            opt = optimizer
    else:
        opt = optimizer
    
    # Compile model
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    
    print(f"\nModel '{model_name}' built successfully!")
    print("Model summary:")
    model.summary()
    
    return model


def create_lstm_model_config(
    lstm_layers: List[int],
    dense_layers: List[int] = None,
    dropout_rates: List[float] = None,
    use_bidirectional: bool = False
) -> List[Dict]:
    """
    Helper function to create LSTM model configurations easily.
    """
    
    if dense_layers is None:
        dense_layers = []
    
    if dropout_rates is None:
        dropout_rates = [0.2] * len(lstm_layers)
    elif len(dropout_rates) != len(lstm_layers):
        dropout_rates = (dropout_rates * len(lstm_layers))[:len(lstm_layers)]
    
    configs = []
    
    # Add LSTM layers
    for i, (units, dropout) in enumerate(zip(lstm_layers, dropout_rates)):
        layer_type = 'bidirectional' if use_bidirectional else 'lstm'
        config = {
            'layer_type': layer_type,
            'units': units,
            'layer_name': f"{'bi_' if use_bidirectional else ''}lstm_{i+1}",
            'dropout': dropout
        }
        
        if use_bidirectional:
            config['rnn_type'] = 'lstm'
            
        configs.append(config)
    
    # Add dense layers
    for i, units in enumerate(dense_layers):
        configs.append({
            'layer_type': 'dense',
            'units': units,
            'layer_name': f"dense_{i+1}",
            'activation': 'relu'
        })
    
    return configs


def create_cnn_lstm_model_config(
    conv_layers: List[Dict],
    lstm_layers: List[int],
    dense_layers: List[int] = None
) -> List[Dict]:
    """
    Helper function to create CNN-LSTM hybrid model configurations.
    
    Parameters:
    -----------
    conv_layers : List[Dict]
        List of conv1d layer configurations
        Each dict should have 'filters', 'kernel_size', and optional parameters
    lstm_layers : List[int]
        List of LSTM layer sizes
    dense_layers : List[int], optional
        List of dense layer sizes
        
    Returns:
    --------
    List[Dict] : Configuration for build_deep_learning_model
    """
    
    if dense_layers is None:
        dense_layers = []
    
    configs = []
    
    # Add convolutional layers
    for i, conv_config in enumerate(conv_layers):
        config = {
            'layer_type': 'conv1d',
            'units': conv_config['filters'],
            'kernel_size': conv_config.get('kernel_size', 3),
            'layer_name': f"conv1d_{i+1}",
            'activation': conv_config.get('activation', 'relu')
        }
        configs.append(config)
        
        # Add pooling if specified
        if conv_config.get('pool_size'):
            configs.append({
                'layer_type': 'maxpooling1d',
                'pool_size': conv_config['pool_size'],
                'layer_name': f"pool_{i+1}"
            })
    
    # Add LSTM layers
    for i, units in enumerate(lstm_layers):
        configs.append({
            'layer_type': 'lstm',
            'units': units,
            'layer_name': f"lstm_{i+1}",
            'dropout': 0.2
        })
    
    # Add dense layers
    for i, units in enumerate(dense_layers):
        configs.append({
            'layer_type': 'dense',
            'units': units,
            'layer_name': f"dense_{i+1}",
            'activation': 'relu'
        })
    
    return configs




