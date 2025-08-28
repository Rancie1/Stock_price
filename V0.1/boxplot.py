import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import List, Optional, Union, Tuple

def plot_stock_boxplot(data: pd.DataFrame, 
                      price_column: str = 'Close',
                      date_column: str = 'Date',
                      window_size: int = 30,
                      num_windows: int = 6,
                      overlap: int = 0,
                      title: Optional[str] = None,
                      figsize: Tuple[int, int] = (12, 8),
                      color_palette: str = 'viridis',
                      show_outliers: bool = True,
                      show_means: bool = True,
                      rotation: int = 45,
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Display stock market financial data using boxplot charts for moving windows of consecutive trading days.
    
    This function is particularly useful for analyzing price distributions over multiple time periods,
    helping to identify trends, volatility patterns, and outliers in stock price movements.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing stock price data with date and price columns
    price_column : str, default 'Close'
        Name of the column containing price data (e.g., 'Close', 'Open', 'High', 'Low', 'Adjusted Close')
    date_column : str, default 'Date'
        Name of the column containing date information
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
        Seaborn color palette name for the boxes ('viridis', 'plasma', 'Set1', etc.)
    show_outliers : bool, default True
        Whether to display outlier points beyond the whiskers
    show_means : bool, default True
        Whether to show mean values as diamond markers on each box
    rotation : int, default 45
        Rotation angle for x-axis labels in degrees
    save_path : str, optional
        File path to save the plot. If None, plot is displayed but not saved
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
        
    Raises:
    -------
    ValueError
        If required columns are missing or if window parameters are invalid
    """
    
    # Input validation
    if data.empty:
        raise ValueError("Input DataFrame is empty")
    
    if price_column not in data.columns:
        raise ValueError(f"Price column '{price_column}' not found in DataFrame")
    
    if date_column not in data.columns:
        raise ValueError(f"Date column '{date_column}' not found in DataFrame")
    
    if window_size <= 0 or num_windows <= 0:
        raise ValueError("Window size and number of windows must be positive integers")
    
    if overlap >= window_size:
        raise ValueError("Overlap must be less than window size")
    
    # Prepare the data
    df = data.copy()
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Sort by date to ensure proper chronological order
    df = df.sort_values(date_column).reset_index(drop=True)
    
    # Remove any NaN values in the price column
    df = df.dropna(subset=[price_column])
    
    # Calculate step size between windows (accounting for overlap)
    step_size = window_size - overlap
    
    # Check if we have enough data
    min_required_data = (num_windows - 1) * step_size + window_size
    if len(df) < min_required_data:
        raise ValueError(f"Not enough data. Need at least {min_required_data} data points, "
                        f"but only have {len(df)}")
    
    # Create windows and extract data for boxplot
    boxplot_data = []
    window_labels = []
    window_stats = []
    
    for i in range(num_windows):
        # Calculate start and end indices for current window
        start_idx = i * step_size
        end_idx = start_idx + window_size
        
        # Extract window data
        window_data = df.iloc[start_idx:end_idx]
        window_prices = window_data[price_column].values
        
        # Store data for boxplot
        boxplot_data.append(window_prices)
        
        # Create window label with date range
        start_date = window_data[date_column].iloc[0].strftime('%Y-%m-%d')
        end_date = window_data[date_column].iloc[-1].strftime('%Y-%m-%d')
        window_labels.append(f'{start_date}\nto\n{end_date}')
        
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
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create boxplot
    box_plot = ax.boxplot(boxplot_data, 
                         labels=window_labels,
                         patch_artist=True,  # Enable filling boxes with colors
                         showfliers=show_outliers,  # Control outlier display
                         showmeans=show_means,  # Show mean markers
                         meanprops={'marker': 'D', 'markerfacecolor': 'red', 'markeredgecolor': 'red', 'markersize': 6})
    
    # Apply color palette to boxes
    colors = sns.color_palette(color_palette, num_windows)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)  # Add some transparency
    
    # Customize the plot appearance
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('Time Windows (Date Ranges)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{price_column} Price', fontsize=12, fontweight='bold')
    
    # Set title
    if title is None:
        symbol_info = f" for {data.get('Symbol', ['Unknown']).iloc[0]}" if 'Symbol' in data.columns else ""
        title = f'Stock Price Distribution Boxplot{symbol_info}\n({window_size}-Day Windows)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=rotation, ha='right')
    
    # Add statistics text box
    stats_text = f"Analysis Summary:\n"
    stats_text += f"• Window Size: {window_size} days\n"
    stats_text += f"• Number of Windows: {num_windows}\n"
    stats_text += f"• Overlap: {overlap} days\n"
    stats_text += f"• Total Data Points: {len(df)}"
    
    # Position text box
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontsize=9)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Display summary statistics
    print("\n" + "="*60)
    print("WINDOW STATISTICS SUMMARY")
    print("="*60)
    for i, (label, stats) in enumerate(zip(window_labels, window_stats)):
        clean_label = label.replace('\n', ' ')
        print(f"\nWindow {i+1}: {clean_label}")
        print(f"  Mean: ${stats['mean']:.2f}")
        print(f"  Median: ${stats['median']:.2f}")
        print(f"  Std Dev: ${stats['std']:.2f}")
        print(f"  Range: ${stats['min']:.2f} - ${stats['max']:.2f}")
        print(f"  Data Points: {stats['count']}")
    
    return fig

# Example usage and demonstration function
def demo_stock_boxplot():
    """
    Demonstration function showing how to use the stock boxplot function
    with sample data generation.
    """
    print("Generating sample stock data for demonstration...")
    
    # Generate sample stock data
    np.random.seed(42)  # For reproducible results
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    # Simulate realistic stock price movement with trend and volatility
    base_price = 100
    price_changes = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
    prices = [base_price]
    
    for change in price_changes[1:]:
        # Add some trending behavior and volatility clustering
        trend_factor = 1 + 0.0002 * len(prices)  # Slight upward trend
        volatility = 0.02 + 0.01 * np.sin(len(prices) / 50)  # Varying volatility
        
        new_price = prices[-1] * (1 + change * trend_factor + np.random.normal(0, volatility * 0.5))
        prices.append(max(new_price, 1))  # Ensure positive prices
    
    # Create DataFrame
    sample_data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Symbol': 'DEMO'
    })
    
    print(f"Sample data created: {len(sample_data)} data points from {dates[0].date()} to {dates[-1].date()}")
    
    # Demonstrate the function with different parameters
    print("\nCreating boxplot visualization...")
    
    # Basic usage
    fig = plot_stock_boxplot(
        data=sample_data,
        price_column='Close',
        window_size=30,  # 30-day windows
        num_windows=8,   # Show 8 windows
        overlap=15,      # 15-day overlap between windows
        title="Sample Stock Price Distribution Analysis",
        show_means=True,
        color_palette='Set2'
    )
    
    plt.show()
    
    print("\nDemonstration completed!")

# Additional utility function for advanced analysis
def compare_multiple_stocks_boxplot(stock_data_dict: dict,
                                   price_column: str = 'Close',
                                   date_column: str = 'Date',
                                   window_size: int = 30,
                                   figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Compare multiple stocks using side-by-side boxplots.
    
    Parameters:
    -----------
    stock_data_dict : dict
        Dictionary with stock symbols as keys and DataFrames as values
    price_column : str
        Column name for price data
    date_column : str
        Column name for date data
    window_size : int
        Size of each time window
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created comparison figure
    """
    
    num_stocks = len(stock_data_dict)
    fig, axes = plt.subplots(1, num_stocks, figsize=figsize, sharey=True)
    
    if num_stocks == 1:
        axes = [axes]
    
    for i, (symbol, data) in enumerate(stock_data_dict.items()):
        # Calculate windows for current stock
        df = data.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column).reset_index(drop=True)
        
        # Determine number of windows based on available data
        max_windows = len(df) // window_size
        num_windows = min(6, max_windows)  # Limit to 6 windows for clarity
        
        if num_windows < 2:
            axes[i].text(0.5, 0.5, f'Insufficient data\nfor {symbol}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            continue
        
        # Extract window data
        boxplot_data = []
        labels = []
        
        for w in range(num_windows):
            start_idx = w * window_size
            end_idx = start_idx + window_size
            window_prices = df.iloc[start_idx:end_idx][price_column].values
            boxplot_data.append(window_prices)
            labels.append(f'Period {w+1}')
        
        # Create boxplot for current stock
        axes[i].boxplot(boxplot_data, labels=labels, patch_artist=True)
        axes[i].set_title(f'{symbol}\n({window_size}-day windows)', fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        
        if i == 0:
            axes[i].set_ylabel(f'{price_column} Price', fontweight='bold')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Run demonstration
    demo_stock_boxplot()