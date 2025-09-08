from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, LSTM, GRU, SimpleRNN, 
    Conv1D, MaxPooling1D, GlobalMaxPooling1D,
    BatchNormalization, Bidirectional, Input
)
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from typing import List, Dict, Union, Optional




def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False, 
                layer_names=None):
    """
    Enhanced version of create_model function with layer_name support.
    
    Parameters:
    -----------
    sequence_length : int
        Length of input sequences (time steps)
    n_features : int
        Number of input features
    units : int
        Number of units in each layer
    cell : keras layer
        Type of RNN cell (LSTM, GRU, SimpleRNN)
    n_layers : int
        Number of RNN layers
    dropout : float
        Dropout rate
    loss : str
        Loss function
    optimizer : str
        Optimizer to use
    bidirectional : bool
        Whether to use bidirectional layers
    layer_names : List[str], optional
        List of custom names for each RNN layer. If None, auto-generates names.
        
    Returns:
    --------
    Sequential model ready for training
    """
    model = Sequential()
    
    # Add input layer first
    model.add(Input(shape=(sequence_length, n_features), name="input_layer"))
    
    # Generate layer names if not provided
    if layer_names is None:
        layer_names = [f'{cell.__name__.lower()}_{i+1}' for i in range(n_layers)]
    elif len(layer_names) != n_layers:
        raise ValueError(f"Number of layer names ({len(layer_names)}) must match n_layers ({n_layers})")
    
    for i in range(n_layers):
        layer_name = layer_names[i]
        
        if i == n_layers - 1:
            # last layer - don't return sequences
            if bidirectional:
                model.add(Bidirectional(
                    cell(units, return_sequences=False, name=f'{layer_name}_cell'),
                    name=layer_name
                ))
            else:
                model.add(cell(units, return_sequences=False, name=layer_name))
        else:
            # first and hidden layers - return sequences
            if bidirectional:
                model.add(Bidirectional(
                    cell(units, return_sequences=True, name=f'{layer_name}_cell'),
                    name=layer_name
                ))
            else:
                model.add(cell(units, return_sequences=True, name=layer_name))
        
        # add dropout after each layer
        model.add(Dropout(dropout, name=f'dropout_{i+1}'))
    
    model.add(Dense(1, activation="linear", name="output_layer"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model


# Example usage
if __name__ == "__main__":
    print("Testing enhanced create_model function...")
    
    # Example 1: Default usage (auto-generated names)
    model1 = create_model(
        sequence_length=60,
        n_features=6,
        units=128,
        n_layers=3
    )
    print("Model 1 created with auto-generated names!")
    print(f"Layer names: {[layer.name for layer in model1.layers]}")
    
    # Example 2: Custom layer names
    model2 = create_model(
        sequence_length=60,
        n_features=6,
        units=64,
        n_layers=2,
        layer_names=['primary_lstm', 'final_lstm'],
        bidirectional=True
    )
    print("\nModel 2 created with custom names!")
    print(f"Layer names: {[layer.name for layer in model2.layers]}")
    
    # Example 3: GRU with custom names
    model3 = create_model(
        sequence_length=60,
        n_features=6,
        units=100,
        cell=GRU,
        n_layers=3,
        layer_names=['gru_encoder', 'gru_processor', 'gru_decoder']
    )
    print("\nModel 3 created with GRU and custom names!")
    print(f"Layer names: {[layer.name for layer in model3.layers]}")