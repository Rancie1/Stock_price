# Task C.7: Sentiment-Based Stock Price Movement Prediction
# Part 1: Data Collection & Preprocessing
# Collects stock data and news data, ensures time alignment

import numpy as np
import pandas as pd
import yfinance as yf
import os
import pickle
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional
import requests
import logging

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load API key from environment
API_KEY = os.getenv('API_KEY')

# Date range for data collection (30 days limit for free NewsAPI)
from datetime import datetime, timedelta
end_date = datetime.now()
start_date = end_date - timedelta(days=30)
START_DATE = start_date.strftime('%Y-%m-%d')
END_DATE = end_date.strftime('%Y-%m-%d')


class SentimentDataCollector:
    """
    Collects and preprocesses stock price data alongside financial news data.
    Ensures data is time-aligned for sentiment analysis.
    """
    
    def __init__(self, company_symbol: str, data_dir: str = '../sentiment_data'):
        """
        Initialize the sentiment data collector.
        
        Parameters:
        -----------
        company_symbol : str
            Stock ticker symbol (e.g., 'CBA.AX', 'AAPL')
        data_dir : str
            Directory to store collected data
        """
        self.company_symbol = company_symbol
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.stock_data = None
        self.news_data = None
        self.combined_data = None
    
        
    def collect_stock_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Download historical stock price data from yfinance.
        
        Parameters:
        -----------
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
            
        Returns:
        --------
        pd.DataFrame
            Stock price data with OHLCV
        """
        logger.info(f"Downloading stock data for {self.company_symbol} from {start_date} to {end_date}")
        
        try:
            data = yf.download(self.company_symbol, start=start_date, end=end_date, progress=False)
            
            # Handle MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Clean data
            data = data.dropna()
            data.reset_index(inplace=True)
            data['Date'] = pd.to_datetime(data['Date']).dt.date
            
            logger.info(f"Downloaded {len(data)} trading days of stock data")
            logger.info(f"Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
            
            self.stock_data = data
            return data
            
        except Exception as e:
            logger.error(f"Error downloading stock data: {e}")
            raise
    
    def collect_news_data(self, api_key: str, start_date: str, end_date: str,
                          query: str = None) -> pd.DataFrame:
        """
        Collect financial news using NewsAPI.
        
        Parameters:
        -----------
        api_key : str
            NewsAPI key from https://newsapi.org
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
        query : str
            Search query (e.g., company name or ticker). If None, uses company_symbol
            
        Returns:
        --------
        pd.DataFrame
            News articles with dates and content
        """
        if not api_key:
            logger.error("API key is required for news collection")
            return None
            
        logger.info(f"Collecting news data from NewsAPI for {query or self.company_symbol}")
        
        try:
            query = query or self.company_symbol
            
            # Build API endpoint
            endpoint = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'apiKey': api_key,
                'from': start_date,
                'to': end_date
            }
            
            # Fetch news
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] != 'ok':
                logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return None
            
            articles = data.get('articles', [])
            logger.info(f"Collected {len(articles)} news articles")
            
            # Convert to DataFrame
            news_df = pd.DataFrame([{
                'date': pd.to_datetime(article['publishedAt']).date(),
                'title': article['title'],
                'description': article['description'],
                'content': article['content'],
                'source': article['source']['name'],
                'url': article['url']
            } for article in articles])
            
            if len(news_df) > 0:
                self.news_data = news_df
                logger.info(f"News date range: {news_df['date'].min()} to {news_df['date'].max()}")
            
            return news_df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading news: {e}")
            return None
        except Exception as e:
            logger.error(f"Error collecting news data: {e}")
            return None
    
    
    def align_data(self, stock_data: pd.DataFrame, news_data: pd.DataFrame) -> pd.DataFrame:
        """
        Align news data with stock data by date.
        Aggregates multiple news articles per day.
        
        Parameters:
        -----------
        stock_data : pd.DataFrame
            Stock price data with 'Date' column
        news_data : pd.DataFrame
            News data with 'date' column
            
        Returns:
        --------
        pd.DataFrame
            Combined stock and news data aligned by date
        """
        logger.info("Aligning news data with stock data by date")
        
        # Ensure both are datetime.date objects
        stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
        news_data['date'] = pd.to_datetime(news_data['date']).dt.date
        
        logger.info(f"Stock data dates: {stock_data['Date'].min()} to {stock_data['Date'].max()}")
        logger.info(f"News data dates: {news_data['date'].min()} to {news_data['date'].max()}")
        logger.info(f"Stock dates available: {sorted(stock_data['Date'].unique().tolist())}")
        logger.info(f"News dates available: {sorted(news_data['date'].unique().tolist())}")
        
        # Group news by date and aggregate
        news_grouped = news_data.groupby('date').agg({
            'title': lambda x: ' | '.join(x),
            'description': lambda x: ' | '.join(x),
            'source': lambda x: ', '.join(x.unique())
        }).reset_index()
        
        news_grouped.rename(columns={'date': 'Date'}, inplace=True)
        
        # Count articles per day
        article_counts = news_data.groupby('date').size().reset_index(name='article_count')
        article_counts.rename(columns={'date': 'Date'}, inplace=True)
        
        # Merge on date
        combined = stock_data.merge(
            news_grouped,
            on='Date',
            how='left'
        )
        
        combined = combined.merge(
            article_counts,
            on='Date',
            how='left'
        )
        
        combined['article_count'] = combined['article_count'].fillna(0).astype(int)
        
        logger.info(f"Combined data shape: {combined.shape}")
        logger.info(f"Date range: {combined['Date'].min()} to {combined['Date'].max()}")
        logger.info(f"Days with news: {(combined['article_count'] > 0).sum()}/{len(combined)}")
        logger.info(f"Total articles aligned: {int(combined['article_count'].sum())}")
        
        self.combined_data = combined
        return combined
    
    def save_data(self, data: pd.DataFrame, filename: str):
        """Save data to pickle file."""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Data saved to {filepath}")
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """Load data from pickle file."""
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Data loaded from {filepath}")
            return data
        else:
            logger.warning(f"File not found: {filepath}")
            return None


# Example usage
if __name__ == "__main__":
    # Check if API key is available
    if not API_KEY:
        logger.error("API_KEY not found in environment variables")
        print("Error: API_KEY not found in .env file")
        print("Get your free API key from: https://newsapi.org/register")
        exit()
    
    # Initialize collector
    collector = SentimentDataCollector('CBA.AX')
    
    # Collect stock data
    logger.info(f"Using date range: {START_DATE} to {END_DATE}")
    print("Collecting stock data...")
    stock_data = collector.collect_stock_data(START_DATE, END_DATE)
    print(f"Stock data: {len(stock_data)} days\n")
    
    # Collect news data
    print("Collecting news data...")
    news_data = collector.collect_news_data(API_KEY, START_DATE, END_DATE, 'Commbank')
    
    if news_data is not None and len(news_data) > 0:
        print(f"News data: {len(news_data)} articles\n")
        
        # Align data
        print("Aligning data...")
        combined_data = collector.align_data(stock_data, news_data)
        print(f"Combined data shape: {combined_data.shape}\n")
        
        # Save combined data
        collector.save_data(combined_data, 'combined_data.pkl')
        print("✓ Data collection and alignment complete!")
    else:
        print("Failed to collect news data. Check API key and connection.")
