import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Optional, Union, Tuple
import warnings
warnings.filterwarnings('ignore')

def plot_stock_boxplot(data: Union[pd.DataFrame, None] = None,
                      company_symbol: str = None,
                      start_date: str = None,
                      end_date: str = None,
                      price_column: str = 'Close',
                      window_size: int = 30,
                      num_windows: int = 6,
                      overlap: int = 0,
                      title: Optional[str] = None,
                      figsize: Tuple[int, int] = (12, 8),
                      color_palette: str = 'viridis',
                      show_outliers: bool = True,
                      show_means: bool = True,
                      rotation: int = 45,
                      show_plot: bool = True,
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Display stock market financial data using boxplot charts for moving windows of consecutive trading days.
    Integrates with yfinance for automatic data download, similar to candlestick chart function.
    
    This function is particularly useful for analyzing price distributions over multiple time periods,
    helping to identify trends, volatility patterns, and outliers in stock price movements.
    
    Parameters:
    -----------
    data : Union[pd.DataFrame, None], optional
        Pre-loaded stock data with OHLC columns, or None to download fresh data
        DataFrame must have columns: ['Open', 'High', 'Low', 'Close'] with DateTime index
        
    company_symbol : str, optional
        Stock symbol to download data for (e.g., 'AAPL', 'CBA.AX', 'MSFT')
        Required if data is None
        
    start_date : str, optional
        Start date for data download in 'YYYY-MM-DD' format
        Required if data is None
        
    end_date : str, optional
        End date for data download in 'YYYY-MM-DD' format
        Required if data is None
        
    price_column : str, default 'Close'
        Name of the column containing price data ('Close', 'Open', 'High', 'Low', 'Adj Close')
        
    window_size : int, default 30
        Number of consecutive trading days in each window (e.g., 30 for ~1 month, 60 for ~3 months)
        
    num_windows : int, default 6
        Number of time windows to display in the boxplot
        
    overlap : int, default 0
        Number of days overlap between consecutive windows (0 = no overlap, window_size-1 = maximum overlap)
        
    title : str, optional
        Custom title for the plot. If None, auto-generates based on parameters
        
    figsize : tuple, default (12, 8)
        Figure size as (width, height) in inches
        
    color_palette : str, default 'viridis'
        Seaborn color palette name for the boxes ('viridis', 'plasma', 'Set1', 'Set2', etc.)
        
    show_outliers : bool, default True
        Whether to display outlier points beyond the whiskers
        
    show_means : bool, default True
        Whether to show mean values as diamond markers on each box
        
    rotation : int, default 45
        Rotation angle for x-axis labels in degrees
        
    show_plot : bool, default True
        Whether to display the chart on screen
        
    save_path : str, optional
        File path to save the plot. If None, plot is displayed but not saved
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
        
    Raises:
    -------
    ValueError
        If required parameters are missing or if window parameters are invalid
    ConnectionError
        If data download from yfinance fails
        
    Examples:
    ---------
    # Basic boxplot with automatic data download
    >>> fig = plot_stock_boxplot(
    ...     company_symbol='AAPL',
    ...     start_date='2023-01-01',
    ...     end_date='2024-01-01',
    ...     window_size=30,
    ...     num_windows=6
    ... )
    
    # Weekly analysis with overlap
    >>> fig = plot_stock_boxplot(
    ...     company_symbol='CBA.AX',
    ...     start_date='2024-01-01',
    ...     end_date='2024-08-27',
    ...     price_column='High',
    ...     window_size=5,
    ...     num_windows=10,
    ...     overlap=2
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
    
    # Validate window parameters
    if window_size <= 0 or num_windows <= 0:
        raise ValueError("Window size and number of windows must be positive integers")
    
    if overlap >= window_size:
        raise ValueError("Overlap must be less than window size")
    
    print("="*60)
    print("STOCK PRICE BOXPLOT GENERATOR")
    print("="*60)
    
    # ================================================================================
    # STEP 2: DATA ACQUISITION (similar to candlestick function)
    # ================================================================================
    
    if data is None:
        # Download fresh data from yfinance
        print(f"Downloading data for {company_symbol} from {start_date} to {end_date}...")
        try:
            # yfinance.download() function with same parameters as candlestick function
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
    # STEP 3: DATA VALIDATION AND CLEANING (similar to candlestick function)
    # ================================================================================
    
    # Handle MultiIndex columns that yfinance sometimes creates
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        print("Flattened MultiIndex columns from yfinance")
    
    # Check if the specified price column exists
    if price_column not in data.columns:
        available_columns = [col for col in data.columns if col in ['Open', 'High', 'Low', 'Close', 'Adj Close']]
        raise ValueError(f"Price column '{price_column}' not found in DataFrame. "
                        f"Available price columns: {available_columns}")
    
    # Ensure the DataFrame index is a proper DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
            print("Converted index to DatetimeIndex for proper time series handling")
        except Exception as e:
            raise ValueError(f"Unable to convert index to datetime format: {str(e)}")
    
    # Sort by date to ensure proper chronological order
    data = data.sort_index()
    
    # Remove any NaN values in the price column
    initial_length = len(data)
    data = data.dropna(subset=[price_column])
    
    if len(data) < initial_length:
        print(f"Removed {initial_length - len(data)} rows with missing {price_column} data")
    
    print(f"Data validation complete. Final shape: {data.shape}")
    print(f"Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    
    # ================================================================================
    # STEP 4: WINDOW CALCULATION AND DATA PREPARATION
    # ================================================================================
    
    # Calculate step size between windows (accounting for overlap)
    step_size = window_size - overlap
    
    # Check if we have enough data
    min_required_data = (num_windows - 1) * step_size + window_size
    if len(data) < min_required_data:
        # Adjust num_windows to fit available data
        max_possible_windows = (len(data) - window_size) // step_size + 1
        if max_possible_windows < 1:
            raise ValueError(f"Not enough data. Need at least {window_size} data points for one window, "
                           f"but only have {len(data)}")
        
        print(f"Warning: Not enough data for {num_windows} windows. Adjusting to {max_possible_windows} windows.")
        num_windows = max_possible_windows
    
    print(f"Creating {num_windows} windows of {window_size} days each with {overlap} days overlap...")
    
    # Create windows and extract data for boxplot
    boxplot_data = []
    window_labels = []
    window_stats = []
    
    for i in range(num_windows):
        # Calculate start and end indices for current window
        start_idx = i * step_size
        end_idx = start_idx + window_size
        
        # Extract window data
        window_data = data.iloc[start_idx:end_idx]
        window_prices = window_data[price_column].values
        
        # Store data for boxplot
        boxplot_data.append(window_prices)
        
        # Create window label with date range
        start_date_str = window_data.index[0].strftime('%Y-%m-%d')
        end_date_str = window_data.index[-1].strftime('%Y-%m-%d')
        
        # Create more compact labels for better readability
        if window_size <= 7:
            # For short windows, show both dates
            window_labels.append(f'{start_date_str}\nto\n{end_date_str}')
        else:
            # For longer windows, show start date and duration
            window_labels.append(f'Window {i+1}\n{start_date_str}\n({window_size} days)')
        
        # Calculate and store statistics for this window
        stats = {
            'mean': np.mean(window_prices),
            'median': np.median(window_prices),
            'std': np.std(window_prices),
            'min': np.min(window_prices),
            'max': np.max(window_prices),
            'count': len(window_prices)
        }
        window_stats.append(stats)
    
    # ================================================================================
    # STEP 5: BOXPLOT GENERATION
    # ================================================================================
    
    print("Generating boxplot visualization...")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create boxplot with enhanced styling
    box_plot = ax.boxplot(boxplot_data, 
                         labels=window_labels,
                         patch_artist=True,  # Enable filling boxes with colors
                         showfliers=show_outliers,  # Control outlier display
                         showmeans=show_means,  # Show mean markers
                         meanprops={'marker': 'D', 'markerfacecolor': 'red', 
                                   'markeredgecolor': 'red', 'markersize': 6},
                         boxprops={'linewidth': 1.5},
                         whiskerprops={'linewidth': 1.5},
                         capprops={'linewidth': 1.5})
    
    # Apply color palette to boxes
    colors = sns.color_palette(color_palette, num_windows)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)  # Add some transparency
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    # Customize the plot appearance
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_xlabel('Time Windows', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{price_column} Price ($)', fontsize=12, fontweight='bold')
    
    # Set title
    if title is None:
        if company_symbol:
            title = f'{company_symbol} Stock Price Distribution Analysis'
        else:
            title = f'Stock Price Distribution Analysis'
        
        title += f'\n{price_column} Price - {window_size}-Day Windows'
        
        if overlap > 0:
            title += f' (Overlap: {overlap} days)'
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=rotation, ha='right')
    
    # Add statistics text box
    stats_text = f"Analysis Summary:\n"
    stats_text += f"• Symbol: {company_symbol or 'N/A'}\n" if company_symbol else ""
    stats_text += f"• Price Column: {price_column}\n"
    stats_text += f"• Window Size: {window_size} days\n"
    stats_text += f"• Number of Windows: {num_windows}\n"
    stats_text += f"• Overlap: {overlap} days\n"
    stats_text += f"• Total Data Points: {len(data)}"
    
    # Position text box in upper left corner
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', 
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
           fontsize=9)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # ================================================================================
    # STEP 6: SAVE AND DISPLAY
    # ================================================================================
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {save_path}")
    
    # Display plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    # ================================================================================
    # STEP 7: SUMMARY OUTPUT
    # ================================================================================
    
    # Display summary statistics
    print("\n" + "="*60)
    print("BOXPLOT SUMMARY")
    print("="*60)
    print(f"Data Points Analyzed: {len(data)}")
    print(f"Date Range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"Price Column: {price_column}")
    print(f"Windows Created: {num_windows}")
    print(f"Window Size: {window_size} days each")
    if overlap > 0:
        print(f"Window Overlap: {overlap} days")
    
    # Overall price statistics
    overall_stats = data[price_column].agg(['min', 'max', 'mean', 'std'])
    print(f"\nOverall {price_column} Price Statistics:")
    print(f"  Range: ${overall_stats['min']:.2f} - ${overall_stats['max']:.2f}")
    print(f"  Mean: ${overall_stats['mean']:.2f}")
    print(f"  Standard Deviation: ${overall_stats['std']:.2f}")
    
    print("\n" + "-"*60)
    print("WINDOW STATISTICS SUMMARY")
    print("-"*60)
    for i, (stats) in enumerate(window_stats):
        print(f"\nWindow {i+1}:")
        print(f"  Mean: ${stats['mean']:.2f}")
        print(f"  Median: ${stats['median']:.2f}")
        print(f"  Std Dev: ${stats['std']:.2f}")
        print(f"  Range: ${stats['min']:.2f} - ${stats['max']:.2f}")
        print(f"  Data Points: {stats['count']}")
    
    print("="*60)
    
    return fig

# ================================================================================
# DEMONSTRATION FUNCTION
# ================================================================================

def demonstrate_boxplot_function():
    """
    Demonstrate the boxplot function with different examples using yfinance data.
    """
    
    print("="*80)
    print("STOCK BOXPLOT FUNCTION DEMONSTRATION")
    print("="*80)
    
    # Example 1: Basic monthly analysis
    print("\nEXAMPLE 1: Monthly Price Distribution Analysis")
    print("-" * 50)
    
    try:
        fig1 = plot_stock_boxplot(
            company_symbol='AAPL',
            start_date='2024-01-01',
            end_date='2024-08-27',
            price_column='Close',
            window_size=30,          # 30-day windows (monthly)
            num_windows=6,           # 6 months
            overlap=0,               # No overlap
            title='Apple Inc. (AAPL) - Monthly Price Distribution',
            color_palette='Set2'
        )
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    # Example 2: Weekly analysis with overlap
    print("\nEXAMPLE 2: Weekly Analysis with Overlap")
    print("-" * 50)
    
    try:
        fig2 = plot_stock_boxplot(
            company_symbol='CBA.AX',
            start_date='2024-06-01',
            end_date='2024-08-27',
            price_column='High',     # Use High prices
            window_size=5,           # 5-day windows (weekly)
            num_windows=8,           # 8 weeks
            overlap=2,               # 2-day overlap
            title='Commonwealth Bank (CBA.AX) - Weekly High Price Analysis',
            color_palette='viridis',
            rotation=30
        )
    except Exception as e:
        print(f"Example 2 failed: {e}")
    
    # Example 3: Quarterly analysis
    print("\nEXAMPLE 3: Quarterly Analysis")
    print("-" * 50)
    
    try:
        fig3 = plot_stock_boxplot(
            company_symbol='MSFT',
            start_date='2023-01-01',
            end_date='2024-01-01',
            price_column='Close',
            window_size=65,          # ~3 months (65 trading days)
            num_windows=4,           # 4 quarters
            overlap=0,               # No overlap
            title='Microsoft (MSFT) - Quarterly Price Distribution',
            color_palette='plasma',
            figsize=(14, 8)
        )
    except Exception as e:
        print(f"Example 3 failed: {e}")

# ================================================================================
# UTILITY FUNCTION FOR MULTIPLE STOCK COMPARISON
# ================================================================================

def compare_multiple_stocks_boxplot(stock_symbols: List[str],
                                   start_date: str,
                                   end_date: str,
                                   price_column: str = 'Close',
                                   window_size: int = 30,
                                   figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Compare multiple stocks using side-by-side boxplots with yfinance data.
    
    Parameters:
    -----------
    stock_symbols : List[str]
        List of stock symbols to compare (e.g., ['AAPL', 'MSFT', 'GOOGL'])
    start_date : str
        Start date for data download in 'YYYY-MM-DD' format
    end_date : str
        End date for data download in 'YYYY-MM-DD' format
    price_column : str
        Column name for price data ('Close', 'Open', 'High', 'Low')
    window_size : int
        Size of each time window in days
    figsize : tuple
        Figure size as (width, height)
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created comparison figure
    """
    
    print(f"Comparing {len(stock_symbols)} stocks: {', '.join(stock_symbols)}")
    
    num_stocks = len(stock_symbols)
    fig, axes = plt.subplots(1, num_stocks, figsize=figsize, sharey=True)
    
    if num_stocks == 1:
        axes = [axes]
    
    for i, symbol in enumerate(stock_symbols):
        try:
            # Download data for current stock
            data = yf.download(symbol, start=start_date, end=end_date, 
                             auto_adjust=True, prepost=True, threads=True)
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            if data.empty or price_column not in data.columns:
                axes[i].text(0.5, 0.5, f'No data\navailable\nfor {symbol}', 
                           ha='center', va='center', transform=axes[i].transAxes,
                           fontsize=12, bbox=dict(boxstyle='round', facecolor='lightcoral'))
                axes[i].set_title(f'{symbol}', fontweight='bold')
                continue
            
            # Calculate number of windows based on available data
            max_windows = len(data) // window_size
            num_windows = min(6, max_windows)  # Limit to 6 windows for clarity
            
            if num_windows < 2:
                axes[i].text(0.5, 0.5, f'Insufficient data\nfor {symbol}\n({len(data)} days)', 
                           ha='center', va='center', transform=axes[i].transAxes,
                           fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow'))
                axes[i].set_title(f'{symbol}', fontweight='bold')
                continue
            
            # Extract window data
            boxplot_data = []
            labels = []
            
            for w in range(num_windows):
                start_idx = w * window_size
                end_idx = start_idx + window_size
                window_prices = data.iloc[start_idx:end_idx][price_column].values
                boxplot_data.append(window_prices)
                labels.append(f'P{w+1}')  # Period 1, 2, 3, etc.
            
            # Create boxplot for current stock
            box_plot = axes[i].boxplot(boxplot_data, labels=labels, patch_artist=True,
                                     showmeans=True, 
                                     meanprops={'marker': 'D', 'markerfacecolor': 'red'})
            
            # Color the boxes
            colors = sns.color_palette('Set3', num_windows)
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            axes[i].set_title(f'{symbol}\n({window_size}-day windows)', fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            
            if i == 0:
                axes[i].set_ylabel(f'{price_column} Price ($)', fontweight='bold')
        
        except Exception as e:
            axes[i].text(0.5, 0.5, f'Error loading\n{symbol}\n{str(e)[:20]}...', 
                       ha='center', va='center', transform=axes[i].transAxes,
                       fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral'))
            axes[i].set_title(f'{symbol} (Error)', fontweight='bold')
    
    plt.suptitle(f'Stock Comparison - {price_column} Price Distribution\n'
                f'{start_date} to {end_date}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return fig

# Main execution
if __name__ == "__main__":
    # Run the demonstration
    demonstrate_boxplot_function()