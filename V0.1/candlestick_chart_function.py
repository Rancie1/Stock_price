# File: candlestick_visualizer.py
# Date: August 27, 2025
# Enhanced Candlestick Chart Function with N-Day Aggregation and Detailed Explanations

import pandas as pd
import numpy as np
import mplfinance as mpf
import yfinance as yf
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

def display_candlestick_chart(
    data: Union[pd.DataFrame, str] = None,
    company_symbol: str = None,
    start_date: str = None,
    end_date: str = None,
    n_days: int = 1,
    chart_type: str = 'candle',
    volume: bool = True,
    moving_averages: Optional[Union[int, List[int], Tuple[int]]] = None,
    style: str = 'yahoo',
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    return_data: bool = False,
    advanced_indicators: bool = False,
    price_panel_height: float = 3.0,
    volume_panel_height: float = 1.0
) -> Optional[Union[pd.DataFrame, Tuple[pd.DataFrame, plt.Figure]]]:
    """
    Display stock market financial data using candlestick charts with n-day aggregation capability.
    
    This function creates professional-quality candlestick charts with extensive customization options,
    including the ability to aggregate multiple trading days into single candlesticks.
    
    Parameters:
    -----------
    data : Union[pd.DataFrame, str], optional
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
        Volume helps confirm price movements and identify accumulation/distribution patterns
        
    moving_averages : Optional[Union[int, List[int], Tuple[int]]], default None
        Moving average periods to overlay on the chart:
        - Single integer: One moving average (e.g., 20)
        - List/Tuple: Multiple moving averages (e.g., [5, 20, 50])
        - None: No moving averages
        
    style : str, default 'yahoo'
        Chart color scheme and styling:
        - 'yahoo': Yahoo Finance style (green up, red down)
        - 'charles': Charles Schwab style
        - 'nightclouds': Dark theme
        - 'sas': SAS style
        - 'starsandstripes': Patriotic theme
        
    figsize : Tuple[int, int], default (12, 8)
        Figure size in inches (width, height)
        Larger sizes provide more detail but use more memory
        
    title : Optional[str], default None
        Custom chart title. If None, auto-generates title based on symbol and date range
        
    save_path : Optional[str], default None
        File path to save the chart image (e.g., 'chart.png', 'analysis.pdf')
        Supports common formats: PNG, PDF, SVG, JPG
        
    show_plot : bool, default True
        Whether to display the chart on screen
        Set to False for batch processing or when only saving files
        
    return_data : bool, default False
        Whether to return the processed data along with the chart
        Useful for further analysis or debugging
        
    advanced_indicators : bool, default False
        Whether to add additional technical indicators:
        - Bollinger Bands
        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence Divergence)
        
    price_panel_height : float, default 3.0
        Relative height of the price chart panel
        Higher values make price chart taller relative to volume
        
    volume_panel_height : float, default 1.0
        Relative height of the volume chart panel
        Higher values make volume chart more prominent
        
    Returns:
    --------
    Optional[Union[pd.DataFrame, Tuple[pd.DataFrame, plt.Figure]]]
        - If return_data=False: None
        - If return_data=True: Processed DataFrame or (DataFrame, Figure) tuple
    
    Raises:
    -------
    ValueError
        If required parameters are missing or invalid
    TypeError
        If data types are incorrect
    ConnectionError
        If unable to download data from yfinance
        
    Examples:
    ---------
    # Basic usage with data download
    >>> display_candlestick_chart(
    ...     company_symbol='AAPL',
    ...     start_date='2023-01-01',
    ...     end_date='2023-12-31'
    ... )
    
    # Weekly candlesticks with moving averages
    >>> display_candlestick_chart(
    ...     company_symbol='CBA.AX',
    ...     start_date='2023-01-01',
    ...     end_date='2023-12-31',
    ...     n_days=5,
    ...     moving_averages=[20, 50],
    ...     style='nightclouds'
    ... )
    
    # Using pre-loaded data with advanced features
    >>> df = yf.download('MSFT', start='2023-01-01', end='2023-12-31')
    >>> display_candlestick_chart(
    ...     data=df,
    ...     n_days=1,
    ...     advanced_indicators=True,
    ...     save_path='msft_analysis.png'
    ... )
    """
    
    # ================================================================================
    # STEP 1: INPUT VALIDATION AND PARAMETER PROCESSING
    # ================================================================================
    
    # Validate that either data is provided OR all download parameters are provided
    if data is None:
        if not all([company_symbol, start_date, end_date]):
            raise ValueError(
                "Either 'data' must be provided, or all of 'company_symbol', "
                "'start_date', and 'end_date' must be specified for data download."
            )
    
    # Validate n_days parameter
    if not isinstance(n_days, int) or n_days < 1:
        raise ValueError("'n_days' must be a positive integer (n_days >= 1)")
    
    # Validate chart_type parameter
    valid_chart_types = ['candle', 'ohlc', 'line']
    if chart_type not in valid_chart_types:
        raise ValueError(f"'chart_type' must be one of {valid_chart_types}")
    
    # Validate and process moving_averages parameter
    if moving_averages is not None:
        # Convert single integer to list for uniform processing
        if isinstance(moving_averages, int):
            moving_averages = [moving_averages]
        elif isinstance(moving_averages, tuple):
            moving_averages = list(moving_averages)
        
        # Validate that all moving average periods are positive integers
        if not all(isinstance(ma, int) and ma > 0 for ma in moving_averages):
            raise ValueError("All moving average periods must be positive integers")
    
    print("="*80)
    print("CANDLESTICK CHART GENERATOR")
    print("="*80)
    
    # ================================================================================
    # STEP 2: DATA ACQUISITION AND VALIDATION
    # ================================================================================
    
    if data is None:
        # Download fresh data from yfinance
        print(f"Downloading data for {company_symbol} from {start_date} to {end_date}...")
        try:
            # yfinance.download() parameters explained:
            # - tickers: Stock symbol(s) to download
            # - start/end: Date range in YYYY-MM-DD format
            # - auto_adjust: Automatically adjust OHLC for splits and dividends
            # - prepost: Include pre and post market hours data
            # - threads: Use threading for faster downloads of multiple symbols
            data = yf.download(
                tickers=company_symbol,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                prepost=True,
                threads=True
            )
            
            if data.empty:
                raise ValueError(f"No data found for symbol '{company_symbol}' in the specified date range")
                
        except Exception as e:
            raise ConnectionError(f"Failed to download data: {str(e)}")
            
        print(f"Successfully downloaded {len(data)} trading days of data")
        
    else:
        # Use provided data
        print("Using provided data...")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("'data' must be a pandas DataFrame")
    
    # ================================================================================
    # STEP 3: DATA STRUCTURE VALIDATION AND STANDARDIZATION
    # ================================================================================
    
    # Handle MultiIndex columns (yfinance sometimes returns MultiIndex)
    if isinstance(data.columns, pd.MultiIndex):
        # Extract the first level (OHLC column names) from MultiIndex
        # This handles cases where yfinance returns columns like ('Close', 'AAPL')
        data.columns = data.columns.get_level_values(0)
        print("Flattened MultiIndex columns")
    
    # Define required columns for OHLC candlestick charts
    required_columns = ['Open', 'High', 'Low', 'Close']
    volume_column = 'Volume'
    
    # Check if all required OHLC columns are present
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}. Available columns: {data.columns.tolist()}")
    
    # Check if Volume column exists (optional but recommended)
    has_volume = volume_column in data.columns
    if volume and not has_volume:
        print("Warning: Volume data not available, disabling volume display")
        volume = False
    
    # Ensure the index is a DatetimeIndex for proper time series handling
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
            print("Converted index to DatetimeIndex")
        except Exception as e:
            raise ValueError(f"Unable to convert index to datetime: {str(e)}")
    
    # Remove any rows with NaN values in required columns
    initial_length = len(data)
    data = data.dropna(subset=required_columns)
    if len(data) < initial_length:
        print(f"Removed {initial_length - len(data)} rows with missing OHLC data")
    
    print(f"Data validation complete. Shape: {data.shape}")
    print(f"Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    
    # ================================================================================
    # STEP 4: N-DAY AGGREGATION LOGIC (ADVANCED FEATURE)
    # ================================================================================
    
    if n_days > 1:
        print(f"Aggregating data into {n_days}-day candlesticks...")
        
        # Create aggregated data using pandas groupby with custom aggregation functions
        # We group by n_days periods and apply different aggregation functions to each column
        
        # Method 1: Using groupby with custom period grouping
        # We create groups of n_days consecutive trading days
        data_copy = data.copy()
        
        # Create group labels by dividing index position by n_days
        # This groups consecutive n_days rows together
        group_labels = np.arange(len(data_copy)) // n_days
        data_copy['group'] = group_labels
        
        # Define aggregation functions for each OHLC column
        agg_functions = {
            'Open': 'first',    # Open of first day in the period
            'High': 'max',      # Highest high in the period
            'Low': 'min',       # Lowest low in the period
            'Close': 'last',    # Close of last day in the period
        }
        
        # Add volume aggregation if available
        if has_volume:
            agg_functions['Volume'] = 'sum'  # Total volume for the period
        
        # Perform the aggregation
        # groupby('group').agg() applies different functions to different columns
        aggregated_data = data_copy.groupby('group').agg(agg_functions)
        
        # Fix the index to represent the last date of each aggregated period
        # This ensures proper time series continuity
        date_groups = data_copy.groupby('group').apply(lambda x: x.index[-1])
        aggregated_data.index = date_groups.values
        
        # Remove any incomplete periods at the end
        # If the last group has fewer than n_days, it represents incomplete data
        last_group_size = len(data_copy[data_copy['group'] == data_copy['group'].max()])
        if last_group_size < n_days:
            aggregated_data = aggregated_data.iloc[:-1]
            print(f"Removed incomplete final period with only {last_group_size} days")
        
        # Replace original data with aggregated data
        data = aggregated_data
        print(f"Aggregation complete. New shape: {data.shape}")
        print(f"Each candlestick now represents {n_days} trading day(s)")
    
    # ================================================================================
    # STEP 5: TECHNICAL INDICATORS CALCULATION (OPTIONAL ADVANCED FEATURES)
    # ================================================================================
    
    indicators_data = {}
    
    # Calculate moving averages if requested
    if moving_averages:
        print(f"Calculating moving averages: {moving_averages}")
        
        for ma_period in moving_averages:
            # Calculate Simple Moving Average (SMA)
            # rolling(window=ma_period) creates a rolling window of the specified size
            # mean() calculates the average of values in each window
            ma_column_name = f'MA_{ma_period}'
            data[ma_column_name] = data['Close'].rolling(window=ma_period).mean()
            
        print("Moving averages calculated successfully")
    
    # Calculate advanced technical indicators if requested
    if advanced_indicators:
        print("Calculating advanced technical indicators...")
        
        # Bollinger Bands calculation
        # Standard deviation bands around a moving average
        bb_period = 20
        bb_std = 2
        
        # Calculate 20-period moving average
        bb_ma = data['Close'].rolling(window=bb_period).mean()
        # Calculate 20-period standard deviation
        bb_std_dev = data['Close'].rolling(window=bb_period).std()
        
        # Upper band = MA + (2 * standard deviation)
        data['BB_Upper'] = bb_ma + (bb_std * bb_std_dev)
        # Lower band = MA - (2 * standard deviation)
        data['BB_Lower'] = bb_ma - (bb_std * bb_std_dev)
        # Middle band is just the moving average
        data['BB_Middle'] = bb_ma
        
        # RSI (Relative Strength Index) calculation
        # Measures the speed and magnitude of recent price changes
        rsi_period = 14
        
        # Calculate price changes
        price_change = data['Close'].diff()
        
        # Separate gains and losses
        gains = price_change.where(price_change > 0, 0)  # Positive changes only
        losses = -price_change.where(price_change < 0, 0)  # Negative changes only (made positive)
        
        # Calculate average gains and losses using exponential weighted moving average
        # This gives more weight to recent values
        avg_gains = gains.ewm(span=rsi_period).mean()
        avg_losses = losses.ewm(span=rsi_period).mean()
        
        # RSI formula: 100 - (100 / (1 + RS))
        # where RS = Average Gains / Average Losses
        rs = avg_gains / avg_losses
        data['RSI'] = 100 - (100 / (1 + rs))
        
        print("Advanced indicators calculated successfully")
    
    # ================================================================================
    # STEP 6: CHART CONFIGURATION AND STYLING
    # ================================================================================
    
    print("Configuring chart appearance...")
    
    # Generate automatic title if not provided
    if title is None:
        if company_symbol:
            title = f"{company_symbol} Stock Price"
        else:
            title = "Stock Price Chart"
            
        # Add aggregation info to title
        if n_days > 1:
            if n_days == 5:
                title += f" (Weekly Candlesticks)"
            elif n_days >= 20 and n_days <= 25:
                title += f" (Monthly Candlesticks)"
            else:
                title += f" ({n_days}-Day Candlesticks)"
    
    # Configure chart panels based on what we want to display
    # addplot list will contain additional plots to overlay on the main chart
    addplot_list = []
    
    # Add moving averages to the main price panel
    if moving_averages:
        # Define colors for different moving averages
        ma_colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink']
        
        for i, ma_period in enumerate(moving_averages):
            ma_column = f'MA_{ma_period}'
            color = ma_colors[i % len(ma_colors)]  # Cycle through colors if more MAs than colors
            
            # Create addplot for moving average line
            # addplot allows us to add additional data series to the chart
            addplot_list.append(
                mpf.make_addplot(
                    data[ma_column],
                    color=color,
                    width=2,
                    secondary_y=False,  # Plot on main y-axis (same as price)
                    panel=0  # Plot on main panel (panel 0 is the price panel)
                )
            )
    
    # Add Bollinger Bands if calculated
    if advanced_indicators and 'BB_Upper' in data.columns:
        # Add upper and lower Bollinger Bands
        addplot_list.append(
            mpf.make_addplot(
                data['BB_Upper'],
                color='gray',
                width=1,
                linestyle='--',  # Dashed line
                alpha=0.7,  # Semi-transparent
                secondary_y=False,
                panel=0
            )
        )
        addplot_list.append(
            mpf.make_addplot(
                data['BB_Lower'],
                color='gray',
                width=1,
                linestyle='--',
                alpha=0.7,
                secondary_y=False,
                panel=0
            )
        )
        
        # Fill area between bands for better visualization
        addplot_list.append(
            mpf.make_addplot(
                data['BB_Middle'],
                color='gray',
                width=1,
                alpha=0.3,
                secondary_y=False,
                panel=0
            )
        )
    
    # Add RSI to a separate panel if calculated
    panel_count = 1  # Start with 1 (main price panel is 0)
    
    if advanced_indicators and 'RSI' in data.columns:
        addplot_list.append(
            mpf.make_addplot(
                data['RSI'],
                color='purple',
                width=2,
                secondary_y=False,
                panel=panel_count,  # RSI gets its own panel
                ylabel='RSI'
            )
        )
        
        # Add RSI reference lines at 30 (oversold) and 70 (overbought)
        # Create horizontal lines using pandas Series filled with constant values
        rsi_oversold = pd.Series([30] * len(data), index=data.index)
        rsi_overbought = pd.Series([70] * len(data), index=data.index)
        
        addplot_list.extend([
            mpf.make_addplot(
                rsi_oversold,
                color='green',
                width=1,
                linestyle='--',
                alpha=0.7,
                secondary_y=False,
                panel=panel_count
            ),
            mpf.make_addplot(
                rsi_overbought,
                color='red',
                width=1,
                linestyle='--',
                alpha=0.7,
                secondary_y=False,
                panel=panel_count
            )
        ])
        
        panel_count += 1  # Increment for next panel
    
    # ================================================================================
    # STEP 7: MAIN CHART CREATION AND RENDERING
    # ================================================================================
    
    print("Rendering candlestick chart...")
    
    try:
        # Configure mplfinance plot parameters
        # This dictionary contains all the styling and configuration options
        plot_kwargs = {
            'type': chart_type,  # Chart type: 'candle', 'ohlc', or 'line'
            'style': style,  # Color scheme and styling
            'title': title,  # Chart title
            'figsize': figsize,  # Figure size in inches
            'volume': volume,  # Whether to show volume panel
            'show_nontrading': False,  # Hide non-trading days (weekends, holidays)
            'warn_too_much_data': 500,  # Warn if more than 500 data points (performance)
        }
        
        # Add moving averages using the built-in mav parameter (alternative to addplot)
        # Note: We're using addplot for more control, but mav is simpler for basic moving averages
        # if moving_averages and len(moving_averages) <= 3:  # mav supports max 3 moving averages
        #     plot_kwargs['mav'] = tuple(moving_averages)
        
        # Add custom plots if any were created
        if addplot_list:
            plot_kwargs['addplot'] = addplot_list
        
        # Configure panel ratios if we have multiple panels
        if advanced_indicators and panel_count > 1:
            # Create panel ratios: price panel gets most space, indicators get less
            panel_ratios = [price_panel_height]  # Main price panel
            
            if volume:
                panel_ratios.append(volume_panel_height)  # Volume panel
            
            # Add ratios for indicator panels
            for _ in range(panel_count - 1):
                panel_ratios.append(1.0)  # Indicator panels
                
            plot_kwargs['panel_ratios'] = tuple(panel_ratios)
        
        # Create the plot using mplfinance
        # mpf.plot() is the main function that creates and displays the chart
        if save_path or not show_plot:
            # If we need to save or not show, we need to get the figure object
            fig, axes = mpf.plot(
                data,
                returnfig=True,  # Return figure object for saving
                **plot_kwargs
            )
            
            # Save the figure if path is provided
            if save_path:
                print(f"Saving chart to: {save_path}")
                fig.savefig(
                    save_path,
                    dpi=300,  # High resolution
                    bbox_inches='tight',  # Remove extra whitespace
                    facecolor='white',  # White background
                    edgecolor='none'  # No border
                )
                print("Chart saved successfully")
            
            # Display the figure if requested
            if show_plot:
                plt.show()
            else:
                plt.close(fig)  # Close figure to free memory
                
        else:
            # Simple plot without saving
            mpf.plot(data, **plot_kwargs)
    
    except Exception as e:
        raise RuntimeError(f"Failed to create chart: {str(e)}")
    
    # ================================================================================
    # STEP 8: SUMMARY STATISTICS AND RETURN VALUES
    # ================================================================================
    
    # Print summary statistics
    print("\n" + "="*80)
    print("CHART SUMMARY")
    print("="*80)
    print(f"Data Points: {len(data)}")
    print(f"Date Range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    
    if n_days > 1:
        print(f"Aggregation: {n_days} trading day(s) per candlestick")
    
    # Calculate and display basic statistics
    price_stats = data['Close'].agg(['min', 'max', 'mean', 'std'])
    print(f"Price Statistics (Close):")
    print(f"  Min: ${price_stats['min']:.2f}")
    print(f"  Max: ${price_stats['max']:.2f}")
    print(f"  Mean: ${price_stats['mean']:.2f}")
    print(f"  Std Dev: ${price_stats['std']:.2f}")
    
    if has_volume:
        volume_stats = data['Volume'].agg(['sum', 'mean'])
        print(f"Volume Statistics:")
        print(f"  Total: {volume_stats['sum']:,.0f}")
        print(f"  Average: {volume_stats['mean']:,.0f}")
    
    print("="*80)
    
    # Return data if requested
    if return_data:
        if save_path and 'fig' in locals():
            return data, fig
        else:
            return data
    
    return None


# ================================================================================
# USAGE EXAMPLES AND DEMONSTRATION
# ================================================================================

def demonstrate_candlestick_features():
    """
    Comprehensive demonstration of the candlestick chart function's capabilities.
    
    This function shows various usage patterns and explains the output of each example.
    """
    
    print("="*100)
    print("CANDLESTICK CHART FUNCTION DEMONSTRATION")
    print("="*100)
    
    # Example 1: Basic daily candlestick chart
    print("\nEXAMPLE 1: Basic Daily Candlesticks with Volume")
    print("-" * 60)
    
    try:
        display_candlestick_chart(
            company_symbol='AAPL',
            start_date='2024-01-01',
            end_date='2024-03-31',
            n_days=1,  # Daily candlesticks
            volume=True,
            title='Apple Inc. (AAPL) - Daily Chart'
        )
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    # Example 2: Weekly aggregated candlesticks with moving averages
    print("\nEXAMPLE 2: Weekly Candlesticks with Moving Averages")
    print("-" * 60)
    
    try:
        display_candlestick_chart(
            company_symbol='CBA.AX',
            start_date='2023-01-01',
            end_date='2024-01-01',
            n_days=5,  # 5-day (weekly) aggregation
            moving_averages=[10, 20],  # 10 and 20 period moving averages
            style='nightclouds',  # Dark theme
            figsize=(14, 10)
        )
    except Exception as e:
        print(f"Example 2 failed: {e}")
    
    # Example 3: Monthly candlesticks with advanced indicators
    print("\nEXAMPLE 3: Monthly Candlesticks with Advanced Technical Indicators")
    print("-" * 60)
    
    try:
        display_candlestick_chart(
            company_symbol='MSFT',
            start_date='2020-01-01',
            end_date='2024-01-01',
            n_days=22,  # ~Monthly aggregation (22 trading days)
            chart_type='candle',
            moving_averages=[12, 26],  # Common MACD parameters
            advanced_indicators=True,  # Enable Bollinger Bands and RSI
            volume=True,
            style='yahoo',
            save_path='msft_monthly_analysis.png'
        )
    except Exception as e:
        print(f"Example 3 failed: {e}")
    
    # Example 4: Using pre-downloaded data
    print("\nEXAMPLE 4: Using Pre-downloaded Data")
    print("-" * 60)
    
    try:
        # Download data separately
        stock_data = yf.download('GOOGL', start='2024-06-01', end='2024-08-27')
        
        # Use the pre-downloaded data
        display_candlestick_chart(
            data=stock_data,
            n_days=1,
            moving_averages=[5, 20, 50],
            volume=True,
            title='Google (GOOGL) - Pre-loaded Data Example',
            return_data=True
        )
    except Exception as e:
        print(f"Example 4 failed: {e}")


# Main execution block
if __name__ == "__main__":
    # Run the demonstration
    demonstrate_candlestick_features()
    
    # Additional example: Quick single chart
    print("\n" + "="*100)
    print("QUICK EXAMPLE: Tesla 3-Month Chart")
    print("="*100)
    
    try:
        display_candlestick_chart(
            company_symbol='TSLA',
            start_date='2024-06-01',
            end_date='2024-08-27',
            n_days=1,
            moving_averages=[20, 50],
            volume=True,
            style='charles'
        )
    except Exception as e:
        print(f"Quick example failed: {e}")
