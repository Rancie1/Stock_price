# File: candlestick_visualizer.py
# Date: August 27, 2025
# Simple Candlestick Chart Function with N-Day Aggregation

import pandas as pd
import numpy as np
import mplfinance as mpf
import yfinance as yf
from typing import Optional, Union, List, Tuple
import warnings
warnings.filterwarnings('ignore')

def display_candlestick_chart(
    data: Union[pd.DataFrame, None] = None,
    company_symbol: str = None,
    start_date: str = None,
    end_date: str = None,
    n_days: int = 1,
    chart_type: str = 'candle',
    volume: bool = True,
    style: str = 'yahoo',
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Display stock market financial data using candlestick charts with n-day aggregation capability.
    
    Parameters:
    -----------
    data : Union[pd.DataFrame, None], optional
        Pre-loaded stock data with OHLC columns, or None to download fresh data
        DataFrame must have columns: ['Open', 'High', 'Low', 'Close', 'Volume'] with DateTime index
        
    company_symbol : str, optional
        Stock symbol to download data for (e.g., 'AAPL', 'CBA.AX', 'MSFT')
        Required if data is None
        
    start_date : str, optional
        Start date for data download in 'YYYY-MM-DD' format
        Required if data is None
        
    end_date : str, optional
        End date for data download in 'YYYY-MM-DD' format
        Required if data is None
        
    n_days : int, default 1
        Number of trading days to aggregate into each candlestick
        - n_days = 1: Daily candlesticks (standard)
        - n_days = 5: Weekly candlesticks (5 trading days)
        - n_days = 22: Monthly candlesticks (approximately 22 trading days per month)
        
    chart_type : str, default 'candle'
        Type of chart to display:
        - 'candle': Traditional candlestick chart with colored bodies
        - 'ohlc': OHLC bar chart with horizontal lines
        - 'line': Simple line chart using closing prices
        
    volume : bool, default True
        Whether to display volume bars below the price chart
        
    style : str, default 'yahoo'
        Chart color scheme and styling:
        - 'yahoo': Yahoo Finance style (green up, red down)
        - 'charles': Charles Schwab style
        - 'nightclouds': Dark theme
        
    figsize : Tuple[int, int], default (12, 8)
        Figure size in inches (width, height)
        
    title : Optional[str], default None
        Custom chart title. If None, auto-generates title based on symbol

        
    show_plot : bool, default True
        Whether to display the chart on screen
        
    Returns:
    --------
    None
    
    Examples:
    ---------
    # Basic daily candlesticks
    >>> display_candlestick_chart(
    ...     company_symbol='AAPL',
    ...     start_date='2023-01-01',
    ...     end_date='2023-12-31'
    ... )
    
    # Weekly candlesticks
    >>> display_candlestick_chart(
    ...     company_symbol='CBA.AX',
    ...     start_date='2023-01-01',
    ...     end_date='2023-12-31',
    ...     n_days=5
    ... )
    """
    
    # ================================================================================
    # STEP 1: INPUT VALIDATION
    # ================================================================================
    
    # Validate that either data is provided OR all download parameters are provided
    if data is None:
        if not all([company_symbol, start_date, end_date]):
            raise ValueError(
                "Either 'data' must be provided, or all of 'company_symbol', "
                "'start_date', and 'end_date' must be specified for data download."
            )
    
    # Validate n_days parameter - must be positive integer
    if not isinstance(n_days, int) or n_days < 1:
        raise ValueError("'n_days' must be a positive integer (n_days >= 1)")
    
    # Validate chart_type parameter - must be one of the supported types
    valid_chart_types = ['candle', 'ohlc', 'line']
    if chart_type not in valid_chart_types:
        raise ValueError(f"'chart_type' must be one of {valid_chart_types}")
    
    print("="*60)
    print("CANDLESTICK CHART GENERATOR")
    print("="*60)
    
    # ================================================================================
    # STEP 2: DATA ACQUISITION
    # ================================================================================
    
    if data is None:
        # Download fresh data from yfinance
        print(f"Downloading data for {company_symbol} from {start_date} to {end_date}...")
        try:
            # yfinance.download() function explained:
            # - tickers: The stock symbol to download (e.g., 'AAPL')
            # - start: Start date in 'YYYY-MM-DD' format
            # - end: End date in 'YYYY-MM-DD' format
            # - auto_adjust: Automatically adjust prices for stock splits and dividends
            # - prepost: Include pre-market and after-hours trading data
            # - threads: Use multi-threading for faster downloads (useful for multiple symbols)
            data = yf.download(
                tickers=company_symbol,
                start=start_date,
                end=end_date,
                auto_adjust=True,  # Adjust for splits/dividends automatically
                prepost=True,      # Include extended hours data
                threads=True       # Use threading for better performance
            )
            
            # Check if download was successful
            if data.empty:
                raise ValueError(f"No data found for symbol '{company_symbol}' in the specified date range")
                
        except Exception as e:
            raise ConnectionError(f"Failed to download data from yfinance: {str(e)}")
            
        print(f"Successfully downloaded {len(data)} trading days of data")
        
    else:
        # Use provided data
        print("Using provided data...")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("'data' parameter must be a pandas DataFrame")
    
    # ================================================================================
    # STEP 3: DATA VALIDATION AND CLEANING
    # ================================================================================
    
    # Handle MultiIndex columns that yfinance sometimes creates
    # When downloading multiple symbols, yfinance creates columns like ('Close', 'AAPL')
    # We need to flatten this to just 'Close' for single symbol analysis
    if isinstance(data.columns, pd.MultiIndex):
        # get_level_values(0) extracts the first level of the MultiIndex
        # This gives us ['Open', 'High', 'Low', 'Close', 'Volume'] instead of 
        # [('Open', 'AAPL'), ('High', 'AAPL'), etc.]
        data.columns = data.columns.get_level_values(0)
        print("Flattened MultiIndex columns from yfinance")
    
    # Define the required columns for candlestick charts
    # OHLC = Open, High, Low, Close - the four essential price points for each trading period
    required_columns = ['Open', 'High', 'Low', 'Close']
    volume_column = 'Volume'
    
    # Check if all required OHLC columns are present in the data
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required OHLC columns: {missing_columns}. "
                        f"Available columns: {data.columns.tolist()}")
    
    # Check if Volume column exists (it's optional but commonly used)
    has_volume = volume_column in data.columns
    if volume and not has_volume:
        print("Warning: Volume column not found in data. Volume display will be disabled.")
        volume = False
    
    # Ensure the DataFrame index is a proper DatetimeIndex
    # This is required for mplfinance to properly handle time series data
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            # pd.to_datetime() can parse many different date formats automatically
            data.index = pd.to_datetime(data.index)
            print("Converted index to DatetimeIndex for proper time series handling")
        except Exception as e:
            raise ValueError(f"Unable to convert index to datetime format: {str(e)}")
    
    # Remove any rows that have NaN (missing) values in the required OHLC columns
    # Missing OHLC data would create gaps in the candlestick chart
    initial_length = len(data)
    data = data.dropna(subset=required_columns)
    
    if len(data) < initial_length:
        print(f"Removed {initial_length - len(data)} rows with missing OHLC data")
    
    print(f"Data validation complete. Final shape: {data.shape}")
    print(f"Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    
    # ================================================================================
    # STEP 4: N-DAY AGGREGATION 
    # ================================================================================
    
    if n_days > 1:
        print(f"Aggregating data into {n_days}-day candlesticks...")
        
        # The n-day aggregation process combines multiple trading days into single candlesticks
        # This is useful for reducing noise and seeing longer-term trends
        
        # Create a copy to avoid modifying the original data
        data_copy = data.copy()
        
        # Create group labels by dividing row positions by n_days
        # np.arange(len(data_copy)) creates [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...]
        # // n_days does integer division to create groups
        # For n_days=5: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, ...]
        # This groups consecutive rows into periods of n_days each
        group_labels = np.arange(len(data_copy)) // n_days
        data_copy['group'] = group_labels
        
        # Define how to aggregate each OHLC column when combining multiple days
        # Each column needs different treatment because they represent different aspects:
        agg_functions = {
            'Open': 'first',   # Opening price = first day's opening price in the period
            'High': 'max',     # Highest price = maximum high reached during any day in the period
            'Low': 'min',      # Lowest price = minimum low reached during any day in the period
            'Close': 'last',   # Closing price = last day's closing price in the period
        }
        
        # Add volume aggregation if volume data is available
        # Volume should be summed because it represents total trading activity
        if has_volume:
            agg_functions['Volume'] = 'sum'  # Total volume = sum of daily volumes
        
        # Perform the aggregation using pandas groupby
        # .groupby('group') groups rows by the group labels we created
        # .agg(agg_functions) applies the specified aggregation function to each column
        aggregated_data = data_copy.groupby('group').agg(agg_functions)
        
        # Fix the datetime index after aggregation
        # After groupby, we lose the original datetime index and get group numbers instead
        # We need to map each group back to a meaningful date
        
        # For each group, take the last (most recent) date as the representative date
        # This lambda function gets the last index value (date) from each group
        date_groups = data_copy.groupby('group').apply(lambda x: x.index[-1])
        
        # Assign these representative dates as the new index
        aggregated_data.index = date_groups.values
        
        # Handle incomplete periods at the end of the data
        # If we have 23 days and n_days=5, the last group only has 3 days
        # We should remove this incomplete period to maintain consistency
        
        # Find the size of the last (most recent) group
        last_group_label = data_copy['group'].max()
        last_group_size = len(data_copy[data_copy['group'] == last_group_label])
        
        # If the last group is incomplete (fewer than n_days), remove it
        if last_group_size < n_days:
            # Remove the last row (incomplete period)
            aggregated_data = aggregated_data.iloc[:-1]
            print(f"Removed incomplete final period with only {last_group_size} day(s)")
        
        # Replace the original data with the aggregated data
        data = aggregated_data
        print(f"Aggregation complete. New shape: {data.shape}")
        print(f"Each candlestick now represents {n_days} trading day(s)")
    
    # ================================================================================
    # STEP 5: CHART GENERATION
    # ================================================================================
    
    print("Generating candlestick chart...")
    
    # Generate automatic title if not provided
    if title is None:
        if company_symbol:
            title = f"{company_symbol} Stock Price"
        else:
            title = "Stock Price Chart"
            
        # Add aggregation information to the title for clarity
        if n_days > 1:
            if n_days == 5:
                title += " (Weekly Candlesticks)"
            elif n_days >= 20 and n_days <= 25:
                title += " (Monthly Candlesticks)"
            else:
                title += f" ({n_days}-Day Candlesticks)"
    
    try:
        # Configure the plot parameters for mplfinance
        # This dictionary contains all the settings for chart appearance and behavior
        plot_kwargs = {
            'type': chart_type,              # Chart type: 'candle', 'ohlc', or 'line'
            'style': style,                  # Visual style: 'yahoo', 'charles', etc.
            'title': title,                  # Chart title
            'figsize': figsize,              # Figure size in inches (width, height)
            'volume': volume,                # Whether to show volume panel below price chart
            'show_nontrading': False,        # Hide weekends and holidays (gaps in data)
            'warn_too_much_data': 500,       # Warn if plotting more than 500 data points
        }
        
        # Handle saving and display options
        if not show_plot:
            # If we need to save the chart or not display it, we need the figure object
            # returnfig=True makes mplfinance return the matplotlib figure object
            # instead of just displaying the chart immediately
            fig, axes = mpf.plot(
                data,                    # The OHLC data to plot
                returnfig=True,         # Return figure object for manipulation
                **plot_kwargs           # Unpack all the plot configuration options
            )
            
        
            
            # Display the chart if requested
            if show_plot:
                # plt.show() displays the matplotlib figure on screen
                import matplotlib.pyplot as plt
                plt.show()
            else:
                # If not displaying, close the figure to free memory
                # This prevents memory leaks when generating many charts
                import matplotlib.pyplot as plt
                plt.close(fig)
                
        else:
            # Simple case: just display the chart without saving
            # mplfinance will handle the display automatically
            mpf.plot(data, **plot_kwargs)
    
    except Exception as e:
        raise RuntimeError(f"Failed to create candlestick chart: {str(e)}")
    
    # ================================================================================
    # STEP 6: SUMMARY OUTPUT
    # ================================================================================
    
    # Display summary statistics about the chart that was created
    print("\n" + "="*60)
    print("CHART SUMMARY")
    print("="*60)
    print(f"Data Points Plotted: {len(data)}")
    print(f"Date Range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    
    if n_days > 1:
        print(f"Aggregation: {n_days} trading day(s) per candlestick")
    
    # Calculate and display basic price statistics
    # .agg() applies multiple aggregation functions at once
    price_stats = data['Close'].agg(['min', 'max', 'mean'])
    print(f"Price Range (Close): ${price_stats['min']:.2f} - ${price_stats['max']:.2f}")
    print(f"Average Close Price: ${price_stats['mean']:.2f}")
    
    # Display volume statistics if volume data is available
    if has_volume and volume:
        volume_stats = data['Volume'].agg(['sum', 'mean'])
        print(f"Total Volume: {volume_stats['sum']:,.0f}")
        print(f"Average Daily Volume: {volume_stats['mean']:,.0f}")
    
    print("="*60)


# ================================================================================
# DEMONSTRATION FUNCTION
# ================================================================================

def demonstrate_candlestick_function():
    """
    Demonstrate the candlestick chart function with different examples.
    """
    
    print("="*80)
    print("CANDLESTICK CHART FUNCTION DEMONSTRATION")
    print("="*80)
    
    # Example 1: Basic daily candlestick chart
    print("\nEXAMPLE 1: Basic Daily Candlesticks")
    print("-" * 50)
    
    try:
        display_candlestick_chart(
            company_symbol='AAPL',
            start_date='2024-06-01',
            end_date='2024-08-27',
            n_days=1,                    # Daily candlesticks
            volume=True,                 # Show volume
            title='Apple Inc. (AAPL) - Daily Chart'
        )
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    # Example 2: Weekly candlesticks
    print("\nEXAMPLE 2: Weekly Candlesticks (5-day aggregation)")
    print("-" * 50)
    
    try:
        display_candlestick_chart(
            company_symbol='CBA.AX',
            start_date='2023-01-01',
            end_date='2024-01-01',
            n_days=5,                    # Weekly aggregation
            chart_type='candle',         
            figsize=(14, 8)
        )
    except Exception as e:
        print(f"Example 2 failed: {e}")
    
    # Example 3: Monthly candlesticks
    print("\nEXAMPLE 3: Monthly Candlesticks (22-day aggregation)")
    print("-" * 50)
    
    try:
        display_candlestick_chart(
            company_symbol='MSFT',
            start_date='2022-01-01',
            end_date='2024-01-01',
            n_days=22,                   # Monthly aggregation (~22 trading days per month)
            chart_type='candle',
            volume=True,
        )
    except Exception as e:
        print(f"Example 3 failed: {e}")


# Main execution
if __name__ == "__main__":
    # Run the demonstration
    demonstrate_candlestick_function()
