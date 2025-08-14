import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Short-Term Momentum and Volatility
    df['EMA_Close_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['Momentum_Factor'] = df['close'] - df['EMA_Close_10']
    
    df['Returns'] = df['close'].pct_change()
    df['StdDev_Returns_10'] = df['Returns'].rolling(window=10).std()
    df['Adaptive_Volatility'] = df['StdDev_Returns_10']

    # Volume Trends
    df['EMA_Volume_10'] = df['volume'].ewm(span=10, adjust=False).mean()
    df['Volume_Trend'] = df['volume'] - df['EMA_Volume_10']

    # Sector Performance (assuming 'Sector_Returns' is a column in the dataframe)
    df['Avg_Sector_Return'] = df['Sector_Returns'].rolling(window=10).mean()
    df['Sector_Performance'] = df['Returns'] - df['Avg_Sector_Return']

    # Dynamic Weights
    df['Liquidity_Factor'] = df['volume'].rolling(window=10).mean()
    max_liquidity = df['Liquidity_Factor'].max()
    df['Adjusted_Momentum'] = df['Momentum_Factor'] * (df['Liquidity_Factor'] / max_liquidity)
    df['Adjusted_Volatility'] = df['Adaptive_Volatility'] * (df['Liquidity_Factor'] / max_liquidity)
    df['Adjusted_Volume_Trend'] = df['Volume_Trend'] * (df['Liquidity_Factor'] / max_liquidity)

    # Composite Alpha Factor
    df['Alpha_Factor'] = (df['Adjusted_Momentum'] + 
                          df['Adjusted_Volume_Trend'] - 
                          df['Adjusted_Volatility'] + 
                          df['Sector_Performance']) / 4

    return df['Alpha_Factor'].dropna()

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
