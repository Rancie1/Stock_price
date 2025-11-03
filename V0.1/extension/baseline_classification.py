# Baseline Stock Price Direction Prediction (Technical Features Only)
# This model uses the same technical features as the sentiment model but excludes all sentiment data
# Purpose: Compare the impact of sentiment analysis on prediction accuracy

import numpy as np
import pandas as pd
import pickle
import os
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Technical indicators calculator (same as sentiment model)."""
    
    def __init__(self):
        pass
    
    def sma(self, data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average."""
        return data.rolling(window=window).mean()
    
    def ema(self, data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average."""
        return data.ewm(span=window).mean()
    
    def rsi(self, data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        # Prevent division by zero
        loss = loss.replace(0, np.nan)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """MACD indicator."""
        ema_fast = self.ema(data, fast)
        ema_slow = self.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def bollinger_bands(self, data: pd.Series, window: int = 20, std_dev: int = 2) -> tuple:
        """Bollinger Bands."""
        sma = self.sma(data, window)
        std = data.rolling(window=window).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    def stochastic_oscillator(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                             k_window: int = 14, d_window: int = 3) -> tuple:
        """Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        denominator = highest_high - lowest_low
        
        # Prevent division by zero
        denominator = denominator.replace(0, np.nan)
        k_percent = 100 * ((close - lowest_low) / denominator)
        k_percent = k_percent.fillna(50)
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    def williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R."""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        denominator = highest_high - lowest_low
        
        # Prevent division by zero
        denominator = denominator.replace(0, np.nan)
        williams_r = -100 * ((highest_high - close) / denominator)
        return williams_r.fillna(-50)
    
    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range."""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=window).mean()


class BaselineFeatureEngineer:
    """Feature engineer for baseline model (technical features only)."""
    
    def __init__(self):
        self.technical_indicators = TechnicalIndicators()
        self.feature_names = []
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators (same as sentiment model)."""
        logger.info("Creating technical indicators...")
        
        # Price-based features
        df['price_change'] = df['Close'].pct_change().fillna(0).replace([np.inf, -np.inf], 1.0)
        df['high_low_ratio'] = (df['High'] / df['Low']).fillna(1).replace([np.inf, -np.inf], 1.0)
        df['close_open_ratio'] = (df['Close'] / df['Open']).fillna(1).replace([np.inf, -np.inf], 1.0)
        
        # Volume features
        df['volume_change'] = df['Volume'].pct_change().fillna(0).replace([np.inf, -np.inf], 1.0)
        df['volume_ratio'] = (df['Volume'] / df['Volume'].rolling(5).mean()).fillna(1).replace([np.inf, -np.inf], 1.0)
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = self.technical_indicators.sma(df['Close'], window)
            df[f'ema_{window}'] = self.technical_indicators.ema(df['Close'], window)
        
        # RSI
        for window in [14, 30]:
            df[f'rsi_{window}'] = self.technical_indicators.rsi(df['Close'], window)
        
        # MACD
        macd, signal, histogram = self.technical_indicators.macd(df['Close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_histogram'] = histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.technical_indicators.bollinger_bands(df['Close'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Stochastic Oscillator
        stoch_k, stoch_d = self.technical_indicators.stochastic_oscillator(df['High'], df['Low'], df['Close'])
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        # Williams %R
        df['williams_r'] = self.technical_indicators.williams_r(df['High'], df['Low'], df['Close'])
        
        # ATR
        df['atr'] = self.technical_indicators.atr(df['High'], df['Low'], df['Close'])
        df['atr_ratio'] = df['atr'] / df['Close']
        
        # Price vs moving averages
        df['close_vs_sma_5'] = (df['Close'] / df['sma_5'] - 1).fillna(0).replace([np.inf, -np.inf], 0)
        df['close_vs_sma_20'] = (df['Close'] / df['sma_20'] - 1).fillna(0).replace([np.inf, -np.inf], 0)
        df['close_vs_ema_20'] = (df['Close'] / df['ema_20'] - 1).fillna(0).replace([np.inf, -np.inf], 0)
        
        # Lagged features
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
            df[f'price_change_lag_{lag}'] = df['price_change'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10]:
            df[f'mean_return_{window}'] = df['price_change'].rolling(window).mean()
            df[f'std_return_{window}'] = df['price_change'].rolling(window).std()
            df[f'skewness_{window}'] = df['price_change'].rolling(window).skew()
            df[f'kurtosis_{window}'] = df['price_change'].rolling(window).kurt()
        
        logger.info(f"Created {len([col for col in df.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']])} technical features")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features for baseline model (no sentiment features)."""
        logger.info("Preparing features for baseline model...")
        
        # Create technical features
        df = self.create_technical_features(df)
        
        # Create target variable (next day price direction)
        df['next_close'] = df['Close'].shift(-1)
        df['price_direction'] = (df['next_close'] > df['Close']).astype(int)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Exclude columns that shouldn't be features
        exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 
                       'title', 'description', 'source', 'url', 'next_close', 'price_direction',
                       'sentiment_score', 'sentiment_label', 'article_count']  # Exclude sentiment features
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_names = feature_cols
        
        # Create feature matrix and handle infinite/large values (only numeric columns)
        X = df[feature_cols].copy()
        
        # Replace infinite values with NaN first (only numeric columns)
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X[numeric_columns] = X[numeric_columns].replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with 0 (only numeric columns)
        X[numeric_columns] = X[numeric_columns].fillna(0)
        
        # Check for extremely large values and cap them (only numeric columns)
        max_abs_value = 1e10  # Cap at 10 billion
        X[numeric_columns] = X[numeric_columns].clip(-max_abs_value, max_abs_value)
        
        # Convert to numpy array (only numeric columns)
        X = X[numeric_columns].values
        
        # Final check for any remaining problematic values
        if np.any(np.isinf(X)) or np.any(np.isnan(X)):
            logger.warning("Found infinite or NaN values in feature matrix, replacing with 0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        y = df['price_direction'].values
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target distribution: {np.bincount(y)}")
        logger.info(f"Feature matrix stats - Min: {X.min():.2f}, Max: {X.max():.2f}, Mean: {X.mean():.2f}")
        
        return X, y, df


class BaselineClassifier:
    """Baseline classifier using only technical features."""
    
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        self.best_model = None
        self.scaler = StandardScaler()
        self.results = {}
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Train all baseline models."""
        logger.info("Training baseline classification models...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0.0
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'auc': auc,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                logger.info(f"{name} - Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                results[name] = {
                    'model': None,
                    'accuracy': 0.0,
                    'auc': 0.0,
                    'predictions': None,
                    'probabilities': None
                }
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        self.best_model = results[best_model_name]['model']
        self.results = results
        
        logger.info(f"Best baseline model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.3f})")
        
        return results
    
    def evaluate_best_model(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Evaluate the best baseline model."""
        if self.best_model is None:
            logger.error("No model available for evaluation")
            return
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.best_model.predict(X_test_scaled)
        
        print("=" * 60)
        print("BASELINE MODEL EVALUATION (TECHNICAL FEATURES ONLY)")
        print("=" * 60)
        print()
        print("=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
        print()
    
    def print_feature_importance(self, feature_names: list, top_n: int = 20) -> None:
        """Print feature importance for tree-based models."""
        if self.best_model is None or not hasattr(self.best_model, 'feature_importances_'):
            print("Feature importance not available for this model type")
            return
        
        importances = self.best_model.feature_importances_
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Top {top_n} Most Important Features:")
        for i, (feature, importance) in enumerate(feature_importance[:top_n], 1):
            print(f"  {i:2d}. {feature:<30} {importance:.4f}")


def main():
    """Main function to run baseline classification."""
    print("=" * 80)
    print("BASELINE STOCK PRICE DIRECTION PREDICTION (TECHNICAL FEATURES ONLY)")
    print("=" * 80)
    print()
    
    # Load data
    logger.info("Loading sentiment data...")
    try:
        with open('../sentiment_data/combined_data.pkl', 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Loaded data shape: {data.shape}")
        logger.info(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    except FileNotFoundError:
        logger.error("Combined data file not found. Run sentiment_data.py first.")
        return
    
    print("Creating baseline features (technical only)...")
    
    # Create features
    engineer = BaselineFeatureEngineer()
    X, y, df = engineer.prepare_features(data)
    
    # Check dataset size
    if len(X) <= 20:
        logger.warning(f"Small dataset detected ({len(X)} samples). Using cross-validation instead of train-test split.")
        
        # Use cross-validation for small datasets
        classifier = BaselineClassifier()
        
        # Scale features
        X_scaled = classifier.scaler.fit_transform(X)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=min(5, len(X)), shuffle=True, random_state=42)
        
        for name, model in classifier.models.items():
            try:
                scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
                logger.info(f"{name} - CV Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
            except Exception as e:
                logger.error(f"Error with {name}: {e}")
        
        print("\n✓ Cross-validation complete!")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Train models
    classifier = BaselineClassifier()
    results = classifier.train_models(X_train, y_train, X_test, y_test)
    
    # Evaluate best model
    classifier.evaluate_best_model(X_test, y_test)
    
    # Print feature importance
    classifier.print_feature_importance(engineer.feature_names)
    
    # Save results
    results_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'models': results,
        'best_model': classifier.best_model,
        'scaler': classifier.scaler,
        'feature_names': engineer.feature_names
    }
    
    with open('../sentiment_data/baseline_results.pkl', 'wb') as f:
        pickle.dump(results_data, f)
    
    print("\n✓ Baseline classification complete! Results saved to sentiment_data/baseline_results.pkl")


if __name__ == "__main__":
    main()
