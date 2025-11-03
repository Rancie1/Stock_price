# Stock Price Prediction with Sentiment Analysis - Usage Guide

## Overview

This extension provides a complete pipeline for predicting stock price direction using technical indicators and news sentiment analysis. It includes data collection, sentiment analysis, feature engineering, model training, and performance comparison tools.

## 📋 Prerequisites

### Required Python Packages
```bash
pip install pandas numpy scikit-learn matplotlib seaborn yfinance requests python-dotenv vaderSentiment
```

### API Keys Setup

1. **Alpha Vantage API Key** (required for news and sentiment data):
   - Sign up at: https://www.alphavantage.co/support/#api-key
   - Get your free API key
   - Add to `.env` file in the project root:

```bash
ALPHA_VANTAGE_API_KEY=your_api_key_here
```

2. **Create `.env` file** in the project root directory:
   ```
   ALPHA_VANTAGE_API_KEY=your_actual_api_key
   ```

## 📁 File Structure

```
extension/
├── sentiment_data.py              # Step 1: Data collection
├── sentiment_analysis.py          # Step 2: Sentiment processing
├── sentiment_classification.py    # Step 3: Model training (with sentiment)
├── baseline_classification.py     # Step 3: Model training (baseline)
├── compare_models.py              # Step 4: Model comparison
└── sentiment_analysis_report.py   # Step 5: Generate analysis report
```

## 🚀 Quick Start Guide

### Step 1: Collect Data

Collect stock price data and news with sentiment from Alpha Vantage.

```bash
cd extension
python sentiment_data.py
```

**What it does:**
- Downloads stock data from Yahoo Finance
- Fetches news articles with pre-calculated sentiment from Alpha Vantage
- Aligns news data with stock data by date
- Saves combined data to `../sentiment_data/combined_data.pkl`

**Configuration:**
- **Ticker Symbol**: Edit line 302 in `sentiment_data.py` to change the stock (default: `'BAC'`)
- **Date Range**: Currently set to 365 days (line 32 in `sentiment_data.py`)

**Output:**
- `../sentiment_data/combined_data.pkl` - Combined stock and news data
- `../sentiment_data/sentiment_data.pkl` - Processed sentiment data

---

### Step 2: Process Sentiment Analysis

Process and enhance sentiment scores (optional if using Alpha Vantage pre-calculated sentiment).

```bash
python sentiment_analysis.py
```

**What it does:**
- Uses Alpha Vantage pre-calculated sentiment if available
- Falls back to VADER sentiment analysis if needed
- Adds sentiment scores to the dataset

**Output:**
- Updated `../sentiment_data/sentiment_data.pkl`

---

### Step 3: Train Models

#### Option A: Train Sentiment-Enhanced Model

Train classification models using both technical and sentiment features.

```bash
python sentiment_classification.py
```

**What it does:**
- Creates 79 features (54 technical + 25 sentiment features)
- Trains 4 models: Random Forest, Gradient Boosting, Logistic Regression, SVM
- Evaluates model performance
- Displays classification report and feature importance

**Output:**
- `../sentiment_data/classification_results.pkl` - Model results and predictions

#### Option B: Train Baseline Model (Technical Features Only)

Train models using only technical indicators (no sentiment).

```bash
python baseline_classification.py
```

**What it does:**
- Creates 54 technical features (same as sentiment model, minus sentiment features)
- Trains the same 4 models for comparison
- Provides baseline performance metrics

**Output:**
- `../sentiment_data/baseline_results.pkl` - Baseline model results

---

### Step 4: Compare Models

Compare the performance of sentiment-enhanced vs baseline models.

```bash
python compare_models.py
```

**What it does:**
- Loads results from both models
- Compares accuracy and AUC metrics
- Shows improvement/decline for each model type
- Analyzes feature importance differences
- Provides summary of sentiment impact

**Output:**
- Console output with comparison table and summary

**Example Output:**
```
📊 MODEL PERFORMANCE COMPARISON
================================================================================
              Model Baseline Accuracy Sentiment Accuracy Accuracy Improvement
      Random Forest             0.500              0.588               +0.088
  Gradient Boosting             0.400              0.569               +0.169
...
```

---

### Step 5: Generate Analysis Report

Generate a comprehensive analysis report with visualizations.

```bash
python sentiment_analysis_report.py
```

**What it does:**
- Loads sentiment data
- Analyzes dataset statistics
- Calculates sentiment-price correlations
- Shows feature importance analysis
- Creates visualizations (if matplotlib is available)
- Provides recommendations

**Output:**
- Console report with statistics and analysis
- Visualizations (price over time, sentiment distribution, correlations)

---

## 🔧 Configuration Options

### Changing Stock Ticker

Edit `sentiment_data.py`, line 302:
```python
ticker_symbol = 'AAPL'  # Change to your desired ticker
```

**Note:** Alpha Vantage works best with US stocks (e.g., 'AAPL', 'MSFT', 'GOOGL'). For Australian stocks (e.g., 'CBA.AX'), use the `.AX` suffix.

### Changing Date Range

Edit `sentiment_data.py`, line 32:
```python
start_date = end_date - timedelta(days=730)  # Change to 2 years
```

### Adjusting Model Parameters

Edit `sentiment_classification.py` or `baseline_classification.py`:

```python
# Modify model hyperparameters in the BaselineClassifier or SentimentClassifier class
self.models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200,  # Increase trees
        random_state=42
    ),
    # ... other models
}
```

---

## 📊 Understanding the Output

### Data Files

1. **`combined_data.pkl`**
   - Columns: Date, Open, High, Low, Close, Volume, title, description, source, sentiment_score, sentiment_label
   - Used for feature engineering

2. **`classification_results.pkl`**
   - Contains: X_train, X_test, y_train, y_test, models, best_model, feature_names
   - Used for analysis and comparison

3. **`baseline_results.pkl`**
   - Same structure as classification_results.pkl
   - Contains baseline model results (no sentiment features)

### Metrics Explained

- **Accuracy**: Percentage of correct predictions
- **AUC (Area Under Curve)**: Measures model's ability to distinguish between classes (0.5 = random, 1.0 = perfect)
- **Feature Importance**: How much each feature contributes to predictions (tree-based models only)

---

## 🐛 Troubleshooting

### Error: "ALPHA_VANTAGE_API_KEY not found"

**Solution:**
1. Create `.env` file in project root (not in `extension/` folder)
2. Add: `ALPHA_VANTAGE_API_KEY=your_key_here`
3. Ensure `python-dotenv` is installed: `pip install python-dotenv`

### Error: "No news articles found"

**Possible causes:**
1. Ticker symbol not recognized by Alpha Vantage (try US stocks like 'AAPL', 'MSFT')
2. Date range too recent (news may not be available yet)
3. API rate limit reached (free tier: 5 calls/minute, 500 calls/day)

**Solution:**
- Try a different, well-known stock ticker
- Use a historical date range (not too recent)
- Wait a few minutes if rate limited

### Error: "FileNotFoundError: combined_data.pkl"

**Solution:**
- Run `sentiment_data.py` first to generate the data file
- Check that files are saved to `../sentiment_data/` directory

### Error: "Small dataset detected"

**Solution:**
- This is a warning, not an error
- The code automatically switches to cross-validation for small datasets
- Collect more data by increasing the date range

### Visualizations Not Showing

**Solution:**
- Ensure matplotlib is installed: `pip install matplotlib seaborn`
- On headless servers, visualizations may not display (code will still run)

---

## 📈 Typical Workflow

For a complete analysis from scratch:

```bash
# 1. Collect data
cd extension
python sentiment_data.py

# 2. (Optional) Process sentiment
python sentiment_analysis.py

# 3. Train sentiment model
python sentiment_classification.py

# 4. Train baseline model
python baseline_classification.py

# 5. Compare results
python compare_models.py

# 6. Generate report
python sentiment_analysis_report.py
```

---

## 🎯 Expected Results

### Typical Performance

- **Baseline Model**: 40-60% accuracy (depending on stock volatility)
- **Sentiment Model**: 45-65% accuracy
- **Improvement**: +2-10% accuracy improvement with sentiment
- **AUC Improvement**: +5-15% typically observed

### Feature Importance Insights

- **Technical features** typically account for 90-97% of importance
- **Sentiment features** typically account for 3-10% of importance
- **Volume lags** and **return statistics** are usually most important

---

## 📝 Notes

1. **Alpha Vantage Free Tier Limits**:
   - 5 API calls per minute
   - 500 API calls per day
   - Plan data collection accordingly

2. **Data Quality**:
   - More data = better model performance
   - Aim for at least 100+ trading days
   - News coverage may be sparse (20-40% of days typically)

3. **Model Selection**:
   - **Random Forest** and **Gradient Boosting** typically perform best with sentiment
   - **Logistic Regression** may overfit with many features
   - Results vary by stock and time period

4. **Sentiment Analysis**:
   - Alpha Vantage provides pre-calculated sentiment (recommended)
   - VADER is used as fallback if Alpha Vantage sentiment unavailable
   - Sentiment scores range from -1 (negative) to +1 (positive)

---

## 🔗 Additional Resources

- **Alpha Vantage API Docs**: https://www.alphavantage.co/documentation/
- **Yahoo Finance (yfinance)**: https://pypi.org/project/yfinance/
- **VADER Sentiment**: https://github.com/cjhutto/vaderSentiment
- **Scikit-learn Documentation**: https://scikit-learn.org/

---

## 💡 Tips for Best Results

1. **Choose liquid stocks** (high trading volume) for better predictions
2. **Use longer date ranges** (1+ years) for more training data
3. **Focus on news-heavy stocks** (finance, tech) for sentiment analysis
4. **Experiment with different tickers** to find optimal performance
5. **Monitor API rate limits** to avoid interruptions
6. **Regularly update data** as new news becomes available

---

*Last Updated: 2025-10-27*
*Extension Version: 1.0*