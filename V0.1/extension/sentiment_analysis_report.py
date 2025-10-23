# sentiment_analysis_report.py
# Comprehensive analysis of sentiment-based stock price prediction

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sentiment_classification import SentimentFeatureEngineer, TechnicalIndicators
import os

def analyze_sentiment_data():
    """Comprehensive analysis of sentiment data and its relationship with stock prices."""
    
    print("="*80)
    print("SENTIMENT-BASED STOCK PRICE ANALYSIS REPORT")
    print("="*80)
    
    # Load sentiment data
    engineer = SentimentFeatureEngineer()
    data = engineer.load_sentiment_data()
    
    if data is None:
        print("❌ Error: Could not load sentiment data. Run sentiment_analysis.py first.")
        return
    
    print(f"\n📊 DATASET OVERVIEW")
    print(f"   • Total days: {len(data)}")
    print(f"   • Date range: {data['Date'].min()} to {data['Date'].max()}")
    print(f"   • Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    
    # Basic sentiment analysis
    if 'sentiment_score' in data.columns:
        print(f"\n📈 SENTIMENT ANALYSIS")
        print(f"   • Average sentiment: {data['sentiment_score'].mean():.3f}")
        print(f"   • Sentiment range: {data['sentiment_score'].min():.3f} to {data['sentiment_score'].max():.3f}")
        print(f"   • Positive days: {(data['sentiment_score'] > 0.1).sum()}")
        print(f"   • Negative days: {(data['sentiment_score'] < -0.1).sum()}")
        print(f"   • Neutral days: {(abs(data['sentiment_score']) <= 0.1).sum()}")
        
        # News volume analysis
        if 'article_count' in data.columns:
            print(f"\n📰 NEWS VOLUME ANALYSIS")
            print(f"   • Total articles: {data['article_count'].sum()}")
            print(f"   • Average articles/day: {data['article_count'].mean():.1f}")
            print(f"   • Days with news: {(data['article_count'] > 0).sum()}")
            print(f"   • Max articles/day: {data['article_count'].max()}")
    
    # Price movement analysis
    data['price_change'] = data['Close'].pct_change()
    data['price_direction'] = (data['price_change'] > 0).astype(int)
    
    print(f"\n💰 PRICE MOVEMENT ANALYSIS")
    print(f"   • Up days: {data['price_direction'].sum()}")
    print(f"   • Down days: {(~data['price_direction'].astype(bool)).sum()}")
    print(f"   • Average daily change: {data['price_change'].mean():.3f}")
    print(f"   • Volatility (std): {data['price_change'].std():.3f}")
    
    # Sentiment vs Price correlation
    if 'sentiment_score' in data.columns:
        correlation = data['sentiment_score'].corr(data['price_change'])
        print(f"\n🔗 SENTIMENT-PRICE CORRELATION")
        print(f"   • Correlation coefficient: {correlation:.3f}")
        
        if abs(correlation) > 0.3:
            print(f"   • {'Strong' if abs(correlation) > 0.7 else 'Moderate'} correlation detected")
        else:
            print(f"   • Weak correlation - sentiment may not be a strong predictor")
    
    # Create visualizations
    create_visualizations(data)
    
    # Feature importance analysis
    analyze_feature_importance()
    
    print(f"\n✅ Analysis complete!")

def create_visualizations(data):
    """Create visualizations for sentiment and price analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Sentiment-Based Stock Price Analysis', fontsize=16)
    
    # 1. Price and Sentiment over time
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    
    ax1.plot(data['Date'], data['Close'], 'b-', linewidth=2, label='Close Price')
    ax1.set_ylabel('Price ($)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    if 'sentiment_score' in data.columns:
        ax1_twin.plot(data['Date'], data['sentiment_score'], 'r-', alpha=0.7, label='Sentiment')
        ax1_twin.set_ylabel('Sentiment Score', color='r')
        ax1_twin.tick_params(axis='y', labelcolor='r')
        ax1_twin.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    ax1.set_title('Price and Sentiment Over Time')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Sentiment distribution
    ax2 = axes[0, 1]
    if 'sentiment_score' in data.columns:
        ax2.hist(data['sentiment_score'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral')
        ax2.set_xlabel('Sentiment Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Sentiment Score Distribution')
        ax2.legend()
    
    # 3. Price changes
    ax3 = axes[1, 0]
    price_changes = data['Close'].pct_change().dropna()
    ax3.hist(price_changes, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No Change')
    ax3.set_xlabel('Daily Price Change (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Daily Price Change Distribution')
    ax3.legend()
    
    # 4. Sentiment vs Price scatter
    ax4 = axes[1, 1]
    if 'sentiment_score' in data.columns:
        scatter = ax4.scatter(data['sentiment_score'], data['Close'].pct_change(), 
                            c=data['Close'].pct_change(), cmap='RdYlGn', alpha=0.7)
        ax4.set_xlabel('Sentiment Score')
        ax4.set_ylabel('Price Change (%)')
        ax4.set_title('Sentiment vs Price Change')
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        plt.colorbar(scatter, ax=ax4, label='Price Change (%)')
    
    plt.tight_layout()
    plt.show()

def analyze_feature_importance():
    """Analyze feature importance from classification results."""
    
    results_file = 'sentiment_data/classification_results.pkl'
    if not os.path.exists(results_file):
        print("\n⚠️  No classification results found. Run sentiment_classification.py first.")
        return
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS")
    
    if 'best_model' in results and results['best_model'] is not None:
        model = results['best_model']
        if hasattr(model, 'feature_importances_'):
            feature_names = results['feature_names']
            importances = model.feature_importances_
            
            # Get top 10 most important features
            indices = np.argsort(importances)[::-1][:10]
            
            print(f"   Top 10 Most Important Features:")
            for i, idx in enumerate(indices, 1):
                print(f"   {i:2d}. {feature_names[idx]:<30} {importances[idx]:.4f}")
            
            # Categorize features
            sentiment_features = [f for f in feature_names if 'sentiment' in f.lower()]
            technical_features = [f for f in feature_names if f not in sentiment_features]
            
            sentiment_importance = sum(importances[i] for i, f in enumerate(feature_names) if 'sentiment' in f.lower())
            technical_importance = sum(importances[i] for i, f in enumerate(feature_names) if 'sentiment' not in f.lower())
            
            print(f"\n   Feature Category Importance:")
            print(f"   • Sentiment features: {sentiment_importance:.3f} ({len(sentiment_features)} features)")
            print(f"   • Technical features: {technical_importance:.3f} ({len(technical_features)} features)")
            
            if sentiment_importance > technical_importance:
                print(f"   ✅ Sentiment features are more important for prediction")
            else:
                print(f"   ⚠️  Technical features are more important than sentiment")
        else:
            print(f"   • Model does not support feature importance analysis")
    else:
        print(f"   • No trained model available for analysis")

def create_prediction_summary():
    """Create a summary of prediction capabilities."""
    
    print(f"\n📋 PREDICTION CAPABILITY SUMMARY")
    
    # Load data
    engineer = SentimentFeatureEngineer()
    data = engineer.load_sentiment_data()
    
    if data is None:
        print("   ❌ No data available for analysis")
        return
    
    # Calculate basic metrics
    data['price_change'] = data['Close'].pct_change()
    data['price_direction'] = (data['price_change'] > 0).astype(int)
    
    up_days = data['price_direction'].sum()
    total_days = len(data) - 1  # Exclude first day (no previous price)
    up_percentage = (up_days / total_days) * 100
    
    print(f"   • Dataset size: {total_days} trading days")
    print(f"   • Up days: {up_days} ({up_percentage:.1f}%)")
    print(f"   • Down days: {total_days - up_days} ({100 - up_percentage:.1f}%)")
    
    # Sentiment analysis
    if 'sentiment_score' in data.columns:
        sentiment_correlation = data['sentiment_score'].corr(data['price_change'])
        print(f"   • Sentiment-price correlation: {sentiment_correlation:.3f}")
        
        if abs(sentiment_correlation) > 0.3:
            print(f"   ✅ Strong sentiment signal detected")
        elif abs(sentiment_correlation) > 0.1:
            print(f"   ⚠️  Moderate sentiment signal")
        else:
            print(f"   ❌ Weak sentiment signal - may need more data or different approach")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    
    if total_days < 30:
        print(f"   • Collect more data (currently {total_days} days, recommend 60+ days)")
    
    if 'sentiment_score' in data.columns:
        if abs(sentiment_correlation) < 0.2:
            print(f"   • Consider different sentiment analysis approach or additional features")
            print(f"   • Try different time windows for sentiment aggregation")
    
    print(f"   • Consider ensemble methods combining multiple models")
    print(f"   • Add more technical indicators or external market data")
    print(f"   • Implement rolling window validation for time series")

if __name__ == "__main__":
    analyze_sentiment_data()
    create_prediction_summary()
