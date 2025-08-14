import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Return
    df['Intraday_Return'] = (df['High'] - df['Low']) / df['Close']
    
    # Calculate Overnight Return
    df['Overnight_Return'] = (df['Open'].shift(-1) - df['Close']) / df['Close']
    
    # Combine Intraday and Overnight Returns
    df['Combined_Return'] = df['Intraday_Return'] + df['Overnight_Return']
    
    # Compute Volume Weighted Average Price
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Volume_Typical_Price'] = df['Typical_Price'] * df['Volume']
    vwap = df['Volume_Typical_Price'].sum() / df['Volume'].sum()
    df['VWAP'] = vwap
    
    # Calculate VWAP Reversal Indicator
    df['Reversal_Indicator'] = df.apply(lambda row: 1 if row['VWAP'] > row['Close'] else -1, axis=1)
    
    # Integrate Reversal Indicator with Combined Return
    df['Integrated_Return'] = df['Combined_Return'] * df['Reversal_Indicator']
    
    # Calculate 20-Day Moving Average of Close Prices
    df['20_MA'] = df['Close'].rolling(window=20).mean()
    
    # Calculate 50-Day Moving Average of Close Prices
    df['50_MA'] = df['Close'].rolling(window=50).mean()
    
    # Calculate Trend Following Indicator
    df['Trend_Indicator'] = df.apply(lambda row: 1 if row['20_MA'] > row['50_MA'] else -1, axis=1)
    
    # Calculate Relative Strength
    df['Relative_Strength'] = df['20_MA'] - df['50_MA']
    
    # Integrate Trend Following and Relative Strength
    df['Trend-Relative_Strength'] = df['Trend_Indicator'] * df['Relative_Strength']
    
    # Incorporate Liquidity Metric
    df['5_day_avg_volume'] = df['Volume'].rolling(window=5).mean()
    df['Liquidity_Indicator'] = df.apply(lambda row: 1 if row['Volume'] > 1.5 * row['5_day_avg_volume'] else 0, axis=1)
    
    # Final Alpha Factor
    df['Final_Alpha_Factor'] = df['Integrated_Return'] * df['Trend-Relative_Strength'] * df['Liquidity_Indicator']
    
    return df['Final_Alpha_Factor']

# Example usage:
# df = pd.DataFrame(...)  # Load your data into a DataFrame
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
