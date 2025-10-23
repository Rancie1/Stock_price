# sentiment_classification.py
# Task C.7: Sentiment-Based Stock Price Movement Prediction
# Part 3: Feature Engineering & Classification for Price Direction Prediction
import numpy as np
import pandas as pd
import pickle
import os
import logging
from typing import Tuple, Dict, List, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators for stock data."""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        # Avoid division by zero
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Neutral RSI when no data
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, window)
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, sma, lower_band
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        # Avoid division by zero
        denominator = highest_high - lowest_low
        k_percent = 100 * ((close - lowest_low) / denominator.replace(0, np.nan))
        k_percent = k_percent.fillna(50)  # Neutral when no range
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        # Avoid division by zero
        denominator = highest_high - lowest_low
        williams_r = -100 * ((highest_high - close) / denominator.replace(0, np.nan))
        return williams_r.fillna(-50)  # Neutral when no range
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()


class SentimentFeatureEngineer:
    """Create features combining technical indicators and sentiment scores."""
    
    def __init__(self, data_dir: str = '../sentiment_data'):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_sentiment_data(self, filename: str = 'sentiment_data.pkl') -> pd.DataFrame:
        """Load sentiment data from pickle file."""
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded sentiment data: {data.shape}")
            return data
        else:
            logger.error(f"Sentiment data file not found: {filepath}")
            return None
    
    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators from stock data."""
        logger.info("Creating technical indicators...")
        
        df = data.copy()
        
        # Price-based features
        df['price_change'] = df['Close'].pct_change().fillna(0)
        df['price_change_abs'] = abs(df['price_change'])
        df['high_low_ratio'] = (df['High'] / df['Low']).replace([np.inf, -np.inf], 1.0)
        df['close_open_ratio'] = (df['Close'] / df['Open']).replace([np.inf, -np.inf], 1.0)
        
        # Volume features
        df['volume_change'] = df['Volume'].pct_change().fillna(0)
        df['volume_sma_20'] = TechnicalIndicators.sma(df['Volume'], 20)
        df['volume_ratio'] = (df['Volume'] / df['volume_sma_20']).replace([np.inf, -np.inf], 1.0)
        
        # Moving averages
        df['sma_5'] = TechnicalIndicators.sma(df['Close'], 5)
        df['sma_10'] = TechnicalIndicators.sma(df['Close'], 10)
        df['sma_20'] = TechnicalIndicators.sma(df['Close'], 20)
        df['sma_50'] = TechnicalIndicators.sma(df['Close'], 50)
        
        df['ema_5'] = TechnicalIndicators.ema(df['Close'], 5)
        df['ema_10'] = TechnicalIndicators.ema(df['Close'], 10)
        df['ema_20'] = TechnicalIndicators.ema(df['Close'], 20)
        
        # Price vs moving averages
        df['close_vs_sma_5'] = (df['Close'] / df['sma_5'] - 1).replace([np.inf, -np.inf], 0)
        df['close_vs_sma_20'] = (df['Close'] / df['sma_20'] - 1).replace([np.inf, -np.inf], 0)
        df['close_vs_ema_20'] = (df['Close'] / df['ema_20'] - 1).replace([np.inf, -np.inf], 0)
        
        # RSI
        df['rsi_14'] = TechnicalIndicators.rsi(df['Close'], 14)
        df['rsi_30'] = TechnicalIndicators.rsi(df['Close'], 30)
        
        # MACD
        macd_line, signal_line, histogram = TechnicalIndicators.macd(df['Close'])
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['Close'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = ((bb_upper - bb_lower) / bb_middle).replace([np.inf, -np.inf], 0)
        df['bb_position'] = ((df['Close'] - bb_lower) / (bb_upper - bb_lower)).replace([np.inf, -np.inf], 0.5)
        
        # Stochastic Oscillator
        k_percent, d_percent = TechnicalIndicators.stochastic_oscillator(df['High'], df['Low'], df['Close'])
        df['stoch_k'] = k_percent
        df['stoch_d'] = d_percent
        
        # Williams %R
        df['williams_r'] = TechnicalIndicators.williams_r(df['High'], df['Low'], df['Close'])
        
        # ATR
        df['atr'] = TechnicalIndicators.atr(df['High'], df['Low'], df['Close'])
        df['atr_ratio'] = (df['atr'] / df['Close']).replace([np.inf, -np.inf], 0)
        
        # Lagged features
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
            df[f'price_change_lag_{lag}'] = df['price_change'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['price_change'].rolling(window).std()
            df[f'mean_return_{window}'] = df['price_change'].rolling(window).mean()
            df[f'skewness_{window}'] = df['price_change'].rolling(window).skew()
            df[f'kurtosis_{window}'] = df['price_change'].rolling(window).kurt()
        
        logger.info(f"Created {len([col for col in df.columns if col not in data.columns])} technical features")
        return df
    
    def create_sentiment_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create sentiment-based features."""
        logger.info("Creating sentiment features...")
        
        df = data.copy()
        
        # Basic sentiment features
        if 'sentiment_score' not in df.columns:
            logger.warning("No sentiment_score column found. Creating dummy sentiment features.")
            df['sentiment_score'] = 0.0
        
        # Sentiment momentum
        df['sentiment_change'] = df['sentiment_score'].diff()
        df['sentiment_ma_5'] = TechnicalIndicators.sma(df['sentiment_score'], 5)
        df['sentiment_ma_10'] = TechnicalIndicators.sma(df['sentiment_score'], 10)
        
        # Sentiment vs price correlation (rolling)
        for window in [5, 10, 20]:
            corr = df['sentiment_score'].rolling(window).corr(df['Close'])
            df[f'sentiment_price_corr_{window}'] = corr.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Sentiment volatility
        df['sentiment_volatility_10'] = df['sentiment_score'].rolling(10).std()
        df['sentiment_volatility_20'] = df['sentiment_score'].rolling(20).std()
        
        # Sentiment extremes
        df['sentiment_extreme_positive'] = (df['sentiment_score'] > 0.5).astype(int)
        df['sentiment_extreme_negative'] = (df['sentiment_score'] < -0.5).astype(int)
        df['sentiment_neutral'] = (abs(df['sentiment_score']) < 0.1).astype(int)
        
        # News volume features
        if 'article_count' in df.columns:
            df['news_volume'] = df['article_count']
            df['news_volume_ma_5'] = TechnicalIndicators.sma(df['article_count'], 5)
            df['news_volume_ma_10'] = TechnicalIndicators.sma(df['article_count'], 10)
            df['news_volume_change'] = df['article_count'].pct_change().fillna(0)
        else:
            df['news_volume'] = 0
            df['news_volume_ma_5'] = 0
            df['news_volume_ma_10'] = 0
            df['news_volume_change'] = 0
        
        # Combined sentiment-price features
        df['sentiment_price_interaction'] = df['sentiment_score'] * df['price_change']
        df['sentiment_volume_interaction'] = df['sentiment_score'] * df['news_volume']
        
        logger.info(f"Created {len([col for col in df.columns if 'sentiment' in col or 'news' in col])} sentiment features")
        return df
    
    def create_target_variable(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create target variable: 1 if next day's close > current close, 0 otherwise."""
        logger.info("Creating target variable...")
        
        df = data.copy()
        df['next_close'] = df['Close'].shift(-1)
        df['price_direction'] = (df['next_close'] > df['Close']).astype(int)
        
        # Remove last row (no next day data)
        df = df[:-1]
        
        logger.info(f"Target distribution: {df['price_direction'].value_counts().to_dict()}")
        return df
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare feature matrix and target vector for classification."""
        logger.info("Preparing features for classification...")
        
        # Create all features
        df = self.create_technical_features(data)
        df = self.create_sentiment_features(df)
        df = self.create_target_variable(df)
        
        # Select feature columns (exclude non-feature columns)
        exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 
                       'title', 'description', 'source', 'url', 'next_close', 'price_direction']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_names = feature_cols
        
        # Create feature matrix and handle infinite/large values
        X = df[feature_cols].copy()
        
        # Replace infinite values with NaN first
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with 0
        X = X.fillna(0)
        
        # Check for extremely large values and cap them
        max_abs_value = 1e10  # Cap at 10 billion
        X = X.clip(-max_abs_value, max_abs_value)
        
        # Convert to numpy array
        X = X.values
        
        # Final check for any remaining problematic values
        if np.any(np.isinf(X)) or np.any(np.isnan(X)):
            logger.warning("Found infinite or NaN values in feature matrix, replacing with 0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        y = df['price_direction'].values
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target distribution: {np.bincount(y)}")
        logger.info(f"Feature matrix stats - Min: {X.min():.2f}, Max: {X.max():.2f}, Mean: {X.mean():.2f}")
        
        return X, y, feature_cols


class SentimentClassifier:
    """Classification model for sentiment-based stock price direction prediction."""
    
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train multiple classification models and return results."""
        logger.info("Training classification models...")
        
        # Check if we have enough data
        if len(X_train) < 10:
            logger.warning(f"Very small training set ({len(X_train)} samples). Results may be unreliable.")
        
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
                
                # Handle AUC calculation for small datasets
                if y_pred_proba is not None and len(np.unique(y_test)) > 1:
                    try:
                        auc = roc_auc_score(y_test, y_pred_proba)
                    except ValueError:
                        auc = 0.5  # Random performance if AUC can't be calculated
                else:
                    auc = 0.5
                
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
                    'auc': 0.5,
                    'predictions': np.zeros(len(y_test)),
                    'probabilities': np.full(len(y_test), 0.5)
                }
        
        # Find best model (filter out failed models)
        valid_results = {k: v for k, v in results.items() if v['model'] is not None}
        if valid_results:
            best_accuracy = max(valid_results.values(), key=lambda x: x['accuracy'])['accuracy']
            best_model_name = max(valid_results.keys(), key=lambda k: valid_results[k]['accuracy'])
            self.best_model = valid_results[best_model_name]['model']
            logger.info(f"Best model: {best_model_name} (Accuracy: {best_accuracy:.3f})")
        else:
            logger.error("All models failed to train!")
            self.best_model = None
        
        return results
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                      feature_names: List[str]) -> None:
        """Evaluate model performance."""
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Classification report
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            self.feature_importance = model.feature_importances_
            self.print_feature_importance(feature_names)
    
    def print_feature_importance(self, feature_names: List[str], top_n: int = 20):
        """Print feature importance."""
        if self.feature_importance is None:
            return
            
        # Get top N features
        indices = np.argsort(self.feature_importance)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importance = self.feature_importance[indices]
        
        print(f"\nTop {top_n} Most Important Features:")
        for i, (feature, importance) in enumerate(zip(top_features, top_importance), 1):
            print(f"  {i:2d}. {feature:<30} {importance:.4f}")


def main():
    """Main execution function."""
    print("="*80)
    print("SENTIMENT-BASED STOCK PRICE DIRECTION PREDICTION")
    print("="*80)
    
    # Initialize feature engineer
    engineer = SentimentFeatureEngineer()
    
    # Load sentiment data
    print("\nLoading sentiment data...")
    data = engineer.load_sentiment_data()
    
    if data is None:
        print("Error: Could not load sentiment data. Run sentiment_analysis.py first.")
        return
    
    print(f"Loaded data shape: {data.shape}")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    
    # Prepare features
    print("\nCreating features...")
    X, y, feature_names = engineer.prepare_features(data)
    
    # Handle small datasets with cross-validation
    if len(X) <= 20:
        print(f"\n⚠️  Small dataset detected ({len(X)} samples). Using cross-validation instead of train-test split.")
        
        # Use cross-validation for small datasets
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        
        # Initialize classifier
        classifier = SentimentClassifier()
        
        # Scale features
        X_scaled = classifier.scaler.fit_transform(X)
        
        # Cross-validation results
        cv_results = {}
        cv = StratifiedKFold(n_splits=min(5, len(X)//2), shuffle=True, random_state=42)
        
        for name, model in classifier.models.items():
            try:
                scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
                cv_results[name] = {
                    'mean_accuracy': scores.mean(),
                    'std_accuracy': scores.std(),
                    'scores': scores
                }
                print(f"{name} - CV Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
            except Exception as e:
                print(f"Error with {name}: {e}")
                cv_results[name] = {'mean_accuracy': 0.0, 'std_accuracy': 0.0, 'scores': []}
        
        # Find best model
        best_cv_model = max(cv_results.keys(), key=lambda k: cv_results[k]['mean_accuracy'])
        print(f"\nBest model (CV): {best_cv_model} (Accuracy: {cv_results[best_cv_model]['mean_accuracy']:.3f})")
        
        # Train best model on full dataset for feature importance
        best_model = classifier.models[best_cv_model]
        best_model.fit(X_scaled, y)
        classifier.best_model = best_model
        
        # Show feature importance if available
        if hasattr(best_model, 'feature_importances_'):
            classifier.feature_importance = best_model.feature_importances_
            classifier.print_feature_importance(feature_names)
        
        print("\n✓ Cross-validation complete!")
        return
    
    # Regular train-test split for larger datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train models
    classifier = SentimentClassifier()
    results = classifier.train_models(X_train, y_train, X_test, y_test)
    
    # Evaluate best model
    print("\n" + "="*60)
    print("BEST MODEL EVALUATION")
    print("="*60)
    classifier.evaluate_model(classifier.best_model, X_test, y_test, feature_names)
    
    # Save results
    results_data = {
        'feature_names': feature_names,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'models': results,
        'best_model': classifier.best_model,
        'scaler': classifier.scaler
    }
    
    with open('sentiment_data/classification_results.pkl', 'wb') as f:
        pickle.dump(results_data, f)
    
    print("\n✓ Classification complete! Results saved to sentiment_data/classification_results.pkl")


if __name__ == "__main__":
    main()