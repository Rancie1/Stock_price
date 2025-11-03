# Task C.7: Sentiment-Based Stock Price Movement Prediction
# Part 2: Simple Sentiment Analysis
# Generates basic sentiment scores from news text at daily level

import numpy as np
import pandas as pd
import pickle
import os
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Simple sentiment analyzer for financial news."""
    
    def __init__(self, data_dir: str = '../sentiment_data'):
        self.data_dir = data_dir
        self.analyzer = SentimentIntensityAnalyzer()
        
    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text using VADER.
        Returns compound score between -1 (negative) and 1 (positive).
        """
        if pd.isna(text) or not text:
            return 0.0
        
        scores = self.analyzer.polarity_scores(str(text))
        return scores['compound']
    
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment scores to combined data."""
        logger.info("Processing sentiment analysis")
        
        df = data.copy()
        
        # Check if we have Alpha Vantage sentiment scores
        if 'sentiment_score' in df.columns and df['sentiment_score'].notna().any():
            logger.info("Using Alpha Vantage pre-calculated sentiment scores")
            # Alpha Vantage scores are already calculated, just ensure they're numeric
            df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce').fillna(0.0)
        else:
            logger.info("Calculating sentiment scores using VADER")
            # Analyze sentiment for each day using VADER
            df['sentiment_score'] = 0.0
            
            for idx, row in df.iterrows():
                # Combine title and description
                text = ""
                if pd.notna(row.get('title')):
                    text += str(row['title']) + " "
                if pd.notna(row.get('description')):
                    text += str(row['description'])
                
                if text.strip():
                    sentiment = self.analyze_sentiment(text)
                    df.at[idx, 'sentiment_score'] = sentiment
        
        logger.info(f"Sentiment analysis complete")
        logger.info(f"Average sentiment: {df['sentiment_score'].mean():.3f}")
        
        return df
    
    def save_data(self, data: pd.DataFrame, filename: str = 'sentiment_data.pkl'):
        """Save sentiment data."""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Data saved to {filepath}")
    
    def load_data(self, filename: str = 'combined_data.pkl') -> pd.DataFrame:
        """Load combined data."""
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Data loaded from {filepath}")
            return data
        else:
            logger.error(f"File not found: {filepath}")
            return None


# Example usage
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    # Load combined data
    print("Loading data...")
    data = analyzer.load_data()
    
    if data is not None:
        print(f"Loaded {len(data)} days of data\n")
        
        # Analyze sentiment
        print("Analyzing sentiment...")
        sentiment_data = analyzer.process_data(data)
        
        # Show results
        print("\nSentiment scores:")
        print(sentiment_data[['Date', 'Close', 'article_count', 'sentiment_score']].head(10))
        
        print(f"\nSentiment summary:")
        print(f"  Average: {sentiment_data['sentiment_score'].mean():.3f}")
        print(f"  Min: {sentiment_data['sentiment_score'].min():.3f}")
        print(f"  Max: {sentiment_data['sentiment_score'].max():.3f}")
        
        # Save
        analyzer.save_data(sentiment_data)
        print("\n✓ Done! Data saved to sentiment_data/sentiment_data.pkl")
    else:
        print("Error: Run sentiment_data.py first to collect data.")
