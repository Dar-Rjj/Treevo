import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Volume-Weighted Price Returns
    df['Open_t1'] = df['open'].shift(-1)
    df['Simple_Returns'] = (df['Open_t1'] - df['close']) / df['close']
    df['Volume_Weighted_Returns'] = df['Simple_Returns'] * df['volume']

    # Identify Volume Surge Days
    df['Volume_Change'] = df['volume'] - df['volume'].shift(1)
    df['Volume_Rolling_Mean'] = df['volume'].rolling(window=5).mean()
    df['Is_Volume_Surge'] = df['volume'] > df['Volume_Rolling_Mean']
    
    # Calculate Volatility
    df['Daily_Returns'] = df['close'].pct_change()
    df['Volatility'] = df['Daily_Returns'].rolling(window=5).std()
    
    # Adjust Volume-Weighted Returns by Volatility
    df['Adjusted_Returns'] = df['Volume_Weighted_Returns'] / df['Volatility']
    
    # Combine Adjusted Returns with Volume Surge Indicator
    surge_factor = 1.5
    df['Combined_Returns'] = df['Adjusted_Returns'] * (surge_factor if df['Is_Volume_Surge'] else 1)

    # Integrate Multi-Timeframe Analysis
    df['Short_Term_MA'] = df['close'].rolling(window=5).mean()
    df['Long_Term_MA'] = df['close'].rolling(window=20).mean()
    df['Trend_Up'] = df['Short_Term_MA'] > df['Long_Term_MA']
    
    upward_trend_factor = 1.2
    downward_trend_factor = 0.8
    df['Final_Alpha_Factor'] = df['Combined_Returns'] * (upward_trend_factor if df['Trend_Up'] else downward_trend_factor)
    
    return df['Final_Alpha_Factor']
