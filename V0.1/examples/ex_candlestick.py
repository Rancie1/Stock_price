# ================================================================================
# DEMONSTRATION FUNCTION
# ================================================================================




from candlestick import display_candlestick_chart

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