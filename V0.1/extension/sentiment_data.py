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

# Load Alpha Vantage API key from environment
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

# Date range for data collection (Alpha Vantage supports historical data)
from datetime import datetime, timedelta
end_date = datetime.now()
start_date = end_date - timedelta(days=365)  # Get 1 year of data
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
    

    def collect_alpha_vantage_news(self, api_key: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Collect financial news with sentiment using Alpha Vantage API.
        
        Parameters:
        -----------
        api_key : str
            Alpha Vantage API key
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
            
        Returns:
        --------
        pd.DataFrame
            News articles with dates, content, and sentiment scores
        """
        if not api_key:
            logger.error("Alpha Vantage API key is required")
            return None
            
        logger.info(f"Collecting news with sentiment from Alpha Vantage for {self.company_symbol}")
        
        try:
            # Convert dates to Alpha Vantage format (YYYYMMDDTHHMM)
            start_datetime = f"{start_date.replace('-', '')}T0000"
            end_datetime = f"{end_date.replace('-', '')}T2359"
            
            # Build Alpha Vantage API endpoint
            endpoint = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': self.company_symbol,
                'time_from': start_datetime,
                'time_to': end_datetime,
                'sort': 'LATEST',
                'limit': 1000,  # Maximum articles
                'apikey': api_key
            }
            
            # Fetch news
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return None
            
            if 'Note' in data:
                logger.warning(f"Alpha Vantage API note: {data['Note']}")
                return None
            
            articles = data.get('feed', [])
            
            if not articles:
                logger.warning("No news articles found from Alpha Vantage")
                return pd.DataFrame()
            
            logger.info(f"Collected {len(articles)} news articles with sentiment")
            
            # Convert to DataFrame
            news_df = pd.DataFrame([{
                'date': pd.to_datetime(article['time_published']).date(),
                'title': article['title'],
                'description': article['summary'],
                'content': article['summary'],
                'source': 'Alpha Vantage',
                'url': article.get('url', ''),
                'sentiment_score': article.get('overall_sentiment_score', 0),
                'sentiment_label': article.get('overall_sentiment_label', 'neutral'),
                'ticker_sentiment': article.get('ticker_sentiment', [{}])[0] if article.get('ticker_sentiment') else {}
            } for article in articles])
            
            # Filter out articles without content
            news_df = news_df[news_df['title'].notna() & (news_df['title'] != '')]
            
            if len(news_df) > 0:
                self.news_data = news_df
                logger.info(f"News date range: {news_df['date'].min()} to {news_df['date'].max()}")
                logger.info(f"Sentiment range: {news_df['sentiment_score'].min():.3f} to {news_df['sentiment_score'].max():.3f}")
            
            return news_df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading news from Alpha Vantage: {e}")
            return None
        except Exception as e:
            logger.error(f"Error collecting Alpha Vantage news data: {e}")
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
        agg_dict = {
            'title': lambda x: ' | '.join(x),
            'description': lambda x: ' | '.join(x),
            'source': lambda x: ', '.join(x.unique())
        }
        
        # Add sentiment columns if they exist (from Alpha Vantage)
        if 'sentiment_score' in news_data.columns:
            agg_dict['sentiment_score'] = 'mean'  # Average sentiment for the day
        if 'sentiment_label' in news_data.columns:
            agg_dict['sentiment_label'] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'neutral'
        
        news_grouped = news_data.groupby('date').agg(agg_dict).reset_index()
        
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
    # Check if Alpha Vantage API key is available
    if not ALPHA_VANTAGE_API_KEY:
        logger.error("ALPHA_VANTAGE_API_KEY not found in environment variables")
        print("Error: ALPHA_VANTAGE_API_KEY not found in .env file")
        print("Get your free API key from: https://www.alphavantage.co/support/#api-key")
        exit()
    
    # Use BAC (Bank of America) - works well with Alpha Vantage
    ticker_symbol = 'BAC'
    
    # Initialize collector
    collector = SentimentDataCollector(ticker_symbol)
    
    # Collect stock data
    logger.info(f"Using date range: {START_DATE} to {END_DATE}")
    print(f"Collecting stock data for {ticker_symbol}...")
    stock_data = collector.collect_stock_data(START_DATE, END_DATE)
    print(f"Stock data: {len(stock_data)} days\n")
    
    # Collect news with sentiment from Alpha Vantage
    print(f"Collecting news with sentiment from Alpha Vantage for {ticker_symbol}...")
    news_data = collector.collect_alpha_vantage_news(ALPHA_VANTAGE_API_KEY, START_DATE, END_DATE)
    
    if news_data is not None and len(news_data) > 0:
        print(f"✅ Alpha Vantage: {len(news_data)} articles with sentiment scores")
        print(f"News data: {len(news_data)} articles\n")
        
        # Align data
        print("Aligning data...")
        combined_data = collector.align_data(stock_data, news_data)
        print(f"Combined data shape: {combined_data.shape}\n")
        
        # Save combined data
        collector.save_data(combined_data, 'combined_data.pkl')
        print("✓ Data collection and alignment complete!")
    else:
        print("❌ Failed to collect news data from Alpha Vantage.")
        print("Check your API key and ticker symbol.")
