# ================================================================================
# DEMONSTRATION FUNCTION
# ================================================================================
from boxplot import plot_stock_boxplot

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


# Main execution
if __name__ == "__main__":
    # Run the demonstration
    demonstrate_boxplot_function()