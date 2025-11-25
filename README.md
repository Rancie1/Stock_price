# Stock Price Prediction System

A comprehensive machine learning system for stock price prediction using deep learning models (LSTM, GRU), ensemble methods (ARIMA + LSTM/GRU), and sentiment analysis. This project provides tools for data collection, preprocessing, model training, prediction, and visualization.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Main Components](#main-components)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Sentiment Analysis Extension](#sentiment-analysis-extension)
- [Requirements](#requirements)

## ✨ Features

- **Deep Learning Models**: LSTM, GRU, and Bidirectional LSTM models for time series prediction
- **Ensemble Methods**: Combines ARIMA (statistical) with deep learning models for improved accuracy
- **Multi-step Prediction**: Predicts stock prices multiple days into the future
- **Multivariate Prediction**: Incorporates multiple features (Open, High, Low, Close, Volume)
- **Sentiment Analysis**: Optional integration of news sentiment data for enhanced predictions
- **Data Visualization**: Candlestick charts, boxplots, and prediction visualizations
- **Flexible Data Processing**: Configurable data loading, scaling, and sequence creation
- **Model Comparison**: Automatic evaluation and comparison of multiple models

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd Stock_price
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **For Sentiment Analysis (Optional):**
   - Sign up for a free API key at [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
   - Create a `.env` file in the project root:
     ```bash
     ALPHA_VANTAGE_API_KEY=your_api_key_here
     ```

## 📁 Project Structure

```
Stock_price/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── V0.1/                              # Main project directory
│   ├── stock_prediction.py           # Main prediction script
│   ├── load_function.py              # Data loading and preprocessing
│   ├── model_builder.py              # Model creation utilities
│   ├── multi_prediction.py           # Multi-step and multivariate prediction
│   ├── ensemble.py                   # ARIMA + LSTM/GRU ensemble
│   ├── ensemble_arima_lstm.py        # ARIMA + LSTM ensemble implementation
│   ├── ensemble_arima_gru.py         # ARIMA + GRU ensemble implementation
│   ├── boxplot.py                    # Boxplot visualization
│   ├── candlestick.py                # Candlestick chart visualization
│   │
│   ├── examples/                     # Usage examples
│   │   ├── ex_load_function.py
│   │   ├── ex_boxplot.py
│   │   ├── ex_candlestick.py
│   │   └── usage_examples.py
│   │
│   ├── extension/                    # Sentiment analysis extension
│   │   ├── Extension_README.md       # Detailed sentiment analysis guide
│   │   ├── sentiment_data.py         # Data collection
│   │   ├── sentiment_analysis.py     # Sentiment processing
│   │   ├── sentiment_classification.py  # Model training with sentiment
│   │   ├── baseline_classification.py   # Baseline model (no sentiment)
│   │   ├── compare_models.py         # Model comparison
│   │   └── sentiment_analysis_report.py # Analysis report generation
│   │
│   ├── stock_data/                    # Stock data storage
│   │   └── CBA.AX_2020-01-01_2024-07-31.pkl
│   │
│   └── sentiment_data/                # Sentiment data storage
│       ├── combined_data.pkl
│       ├── sentiment_data.pkl
│       ├── classification_results.pkl
│       └── baseline_results.pkl
│
├── stock_data/                        # Root-level stock data
└── sentiment_data/                    # Root-level sentiment data
```

## 🎯 Quick Start

### Basic Stock Prediction

Run the main prediction script:

```bash
cd V0.1
python stock_prediction.py
```

**Configuration** (edit at the top of `stock_prediction.py`):
```python
COMPANY = 'CBA.AX'           # Stock symbol
TRAIN_START = '2020-01-01'   # Training start date
TRAIN_END = '2024-07-31'     # Training end date
PREDICTION_DAYS = 60          # Sequence length for prediction
```

**Output:**
- Model training progress and metrics (RMSE, MAE)
- Model comparison table
- Prediction plots
- Next day price prediction
- Multi-step predictions (5 days ahead)
- Multivariate predictions
- Combined multivariate-multistep analysis

## 🔧 Main Components

### 1. Data Loading (`load_function.py`)

Loads and preprocesses stock data from Yahoo Finance:

```python
from load_function import load_and_process_stock_data

X_train, X_test, y_train, y_test, metadata = load_and_process_stock_data(
    company_symbol='CBA.AX',
    start_date='2020-01-01',
    end_date='2024-07-31',
    n_steps=60,
    target_column='Close',
    create_sequences=True,
    save_locally=True,
    load_locally=True,
    split_method='ratio',
    split_value=0.8
)
```

**Features:**
- Automatic data download from Yahoo Finance
- Multiple feature support (Open, High, Low, Close, Volume)
- Data scaling (MinMaxScaler)
- Sequence creation for time series models
- Train/test splitting (ratio, date, or random)
- Local caching of processed data

### 2. Model Building (`model_builder.py`)

Creates customizable deep learning models:

```python
from model_builder import create_model
from tensorflow.keras.layers import LSTM, GRU

model = create_model(
    sequence_length=60,
    n_features=5,
    units=50,
    n_layers=2,
    dropout=0.2,
    cell=LSTM,  # or GRU
    bidirectional=False,
    layer_names=['encoder_lstm', 'decoder_lstm']
)
```

**Supported Models:**
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Bidirectional LSTM/GRU
- Configurable depth and dropout

### 3. Multi-step Prediction (`multi_prediction.py`)

Predicts multiple days into the future:

```python
from multi_prediction import predict_multistep_prices

predictions = predict_multistep_prices(
    model=trained_model,
    last_sequence=last_sequence,
    n_steps=60,
    k_days=5,  # Predict 5 days ahead
    target_scaler=scaler,
    target_feature_idx=0
)
```

### 4. Ensemble Models (`ensemble.py`, `ensemble_arima_lstm.py`, `ensemble_arima_gru.py`)

Combines ARIMA statistical models with deep learning:

```python
from ensemble_arima_lstm import EnsembleStockPredictor

ensemble = EnsembleStockPredictor(
    arima_order=(5, 1, 0),
    ensemble_weights={'arima': 0.3, 'lstm': 0.7}
)

ensemble.fit(train_data, lstm_model)
predictions = ensemble.predict(test_data)
```

## 📊 Usage Examples

### Example 1: Basic Prediction

```python
from load_function import load_and_process_stock_data
from model_builder import create_model
from stock_prediction import train_and_evaluate_model

# Load data
X_train, X_test, y_train, y_test, metadata = load_and_process_stock_data(
    company_symbol='AAPL',
    start_date='2020-01-01',
    end_date='2024-01-01',
    n_steps=60
)

# Create model
model = create_model(
    sequence_length=X_train.shape[1],
    n_features=X_train.shape[2],
    units=50,
    n_layers=2
)

# Train and evaluate
results = train_and_evaluate_model(
    model, "LSTM Model",
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    metadata=metadata
)
```

### Example 2: Visualization

```python
# Candlestick chart
from candlestick import plot_candlestick
plot_candlestick('AAPL', '2024-01-01', '2024-07-31')

# Boxplot
from boxplot import plot_stock_boxplot
plot_stock_boxplot('AAPL', '2024-01-01', '2024-07-31')
```

See `V0.1/examples/` for more detailed examples.

## ⚙️ Configuration

### Stock Prediction Configuration

Edit `V0.1/stock_prediction.py`:

```python
COMPANY = 'CBA.AX'              # Stock symbol (use .AX for Australian stocks)
TRAIN_START = '2020-01-01'      # Training data start date
TRAIN_END = '2024-07-31'        # Training data end date
PREDICTION_DAYS = 60            # Sequence length (time steps)
```

### Model Configuration

Adjust model parameters in `stock_prediction.py`:

```python
model = create_model(
    sequence_length=60,
    n_features=5,
    units=50,           # Number of units in each layer
    n_layers=2,         # Number of LSTM/GRU layers
    dropout=0.2,        # Dropout rate
    epochs=20,          # Training epochs
    batch_size=16       # Batch size
)
```

## 📈 Sentiment Analysis Extension

The project includes an optional sentiment analysis extension that incorporates news sentiment data into predictions.

### Quick Start for Sentiment Analysis

1. **Collect data:**
   ```bash
   cd V0.1/extension
   python sentiment_data.py
   ```

2. **Train sentiment-enhanced model:**
   ```bash
   python sentiment_classification.py
   ```

3. **Compare with baseline:**
   ```bash
   python compare_models.py
   ```

4. **Generate report:**
   ```bash
   python sentiment_analysis_report.py
   ```

For detailed instructions, see `V0.1/extension/Extension_README.md`.

**Features:**
- Automatic news data collection from Alpha Vantage
- Sentiment scoring using VADER or Alpha Vantage pre-calculated sentiment
- Feature engineering with 79 features (54 technical + 25 sentiment)
- Model comparison (sentiment vs. baseline)
- Performance analysis and reporting

## 📦 Requirements

Key dependencies (see `requirements.txt` for complete list):

- **Deep Learning**: `tensorflow>=2.19.0`, `keras>=3.11.1`
- **Data Processing**: `pandas>=2.3.1`, `numpy>=2.1.3`
- **Machine Learning**: `scikit-learn>=1.7.1`
- **Time Series**: `statsmodels` (for ARIMA)
- **Data Sources**: `yfinance>=0.2.65`, `pandas-datareader>=0.10.0`
- **Visualization**: `matplotlib>=3.10.5`, `mplfinance>=0.12.10b0`
- **Sentiment Analysis**: `vaderSentiment` (optional, for sentiment extension)
- **Utilities**: `python-dotenv>=1.1.1`, `requests>=2.32.4`

## 📝 Output Description

When running `stock_prediction.py`, you'll see:

1. **Data Loading**: Information about loaded data, features, and sequence shapes
2. **Model Training**: Training progress for each model with loss metrics
3. **Model Performance**: RMSE and MAE for each model
4. **Model Comparison**: Table comparing all trained models
5. **Best Model Selection**: Identification of the best-performing model
6. **Visualization**: Plot showing actual vs. predicted prices
7. **Next Day Prediction**: Predicted price for the next trading day
8. **Multi-step Prediction**: Predictions for multiple days ahead
9. **Multivariate Prediction**: Predictions using multiple features
10. **Combined Analysis**: Results from combined multivariate-multistep approach

## 🎓 Model Types

### 1. Simple LSTM
- 2-layer LSTM architecture
- 50 units per layer
- 20% dropout
- Good baseline for time series prediction

### 2. Deep LSTM
- 4-layer LSTM architecture
- 64 units per layer
- 30% dropout
- Better for complex patterns

### 3. Bidirectional LSTM
- 2-layer bidirectional LSTM
- 128 units per layer
- Captures patterns in both directions

### 4. GRU Model
- 3-layer GRU architecture
- 100 units per layer
- Faster training than LSTM

### 5. Ensemble Models
- ARIMA + LSTM/GRU combination
- Weighted averaging of predictions
- Improved robustness

## 🔍 Troubleshooting

### Common Issues

1. **Data Download Errors:**
   - Check internet connection
   - Verify stock symbol format (e.g., 'CBA.AX' for Australian stocks)
   - Ensure date range is valid

2. **Memory Errors:**
   - Reduce `PREDICTION_DAYS` (sequence length)
   - Reduce batch size
   - Use fewer features

3. **Model Training Issues:**
   - Ensure sufficient training data (at least 100+ days)
   - Check that data is properly scaled
   - Verify sequence shapes match model input

4. **Sentiment Analysis:**
   - Ensure `.env` file exists with valid API key
   - Check Alpha Vantage API rate limits (5 calls/minute, 500/day)
   - Some stocks may have limited news coverage

## 📚 Additional Resources

- **Examples**: See `V0.1/examples/` for detailed usage examples
- **Sentiment Analysis**: See `V0.1/extension/Extension_README.md` for comprehensive sentiment analysis guide
- **Data Format**: Stock data is stored as pickle files in `stock_data/` directory
- **Model Persistence**: Models can be saved/loaded using Keras `model.save()` and `load_model()`

## 📄 License

This project is for educational purposes as part of an Intelligent Systems course assignment.

## 👤 Author

Project C - Stock Price Prediction System

---

**Note**: Stock price prediction is inherently uncertain. This system is for educational and research purposes only. Always conduct thorough analysis and consult financial advisors before making investment decisions.
