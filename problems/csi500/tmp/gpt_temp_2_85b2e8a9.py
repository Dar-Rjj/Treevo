import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Define the lookback periods
    sma_short_period = 20
    sma_long_period = 60
    atr_period = 14
    vol_adj_volatility_period = 20
    daily_turnover_period = 20
    
    # Calculate Simple Moving Average (SMA) of Close Prices
    df['SMA_Short'] = df['close'].rolling(window=sma_short_period).mean()
    df['SMA_Long'] = df['close'].rolling(window=sma_long_period).mean()
    
    # Compute Volume-Adjusted Volatility
    df['High_Low_Diff'] = df['high'] - df['low']
    df['Vol_Weighted_High_Low'] = df['High_Low_Diff'] * df['volume']
    df['Vol_Adjusted_Volatility'] = df['Vol_Weighted_High_Low'].rolling(window=vol_adj_volatility_period).mean()
    
    # Enhance Volatility Measures
    df['True_Range'] = df[['high', 'low']].apply(lambda x: max(x[0] - x[1], abs(x[0] - df['close'].shift(1)), abs(x[1] - df['close'].shift(1))), axis=1)
    df['ATR'] = df['True_Range'].rolling(window=atr_period).mean()
    
    # Compute Price Momentum
    df['Price_Momentum'] = (df['close'] - df['SMA_Short']) / df['close'].rolling(window=sma_short_period).mean()
    
    # Incorporate Additional Price Change Metrics
    df['Percentage_Change'] = df['close'].pct_change(periods=sma_short_period)
    df['High_Low_Range'] = df['high'] - df['low']
    
    # Consider Market Trend Alignment
    df['Market_Trend'] = df['SMA_Short'] > df['SMA_Long']
    df['Trend_Indicator'] = df['Market_Trend'].apply(lambda x: 1 if x else -1)
    
    # Incorporate Liquidity Measures
    df['Daily_Turnover'] = df['volume'] * df['close']
    df['Rolling_Daily_Turnover'] = df['Daily_Turnover'].rolling(window=daily_turnover_period).mean()
    df['Liquidity_Factor'] = df['Rolling_Daily_Turnover'] / df['Daily_Turnover'].rolling(window=daily_turnover_period).std()
    
    # Final Alpha Factor
    weights = {
        'Price_Momentum': 0.3,
        'Enhanced_Volatility': 0.2,
        'Additional_Price_Change': 0.1,
        'Market_Trend': 0.2,
        'Liquidity': 0.2
    }
    
    df['Alpha_Factor'] = (
        weights['Price_Momentum'] * df['Price_Momentum'] +
        weights['Enhanced_Volatility'] * df['ATR'] / df['Vol_Adjusted_Volatility'] +
        weights['Additional_Price_Change'] * df['Percentage_Change'] +
        weights['Market_Trend'] * df['Trend_Indicator'] +
        weights['Liquidity'] * df['Liquidity_Factor']
    )
    
    # Adjust Weights Based on Market Trend and Liquidity
    df['Alpha_Factor'] = df.apply(
        lambda row: (
            row['Alpha_Factor'] * 
            (1 + 0.1 * row['Trend_Indicator']) * 
            (1 + 0.1 * (row['Liquidity_Factor'] - df['Liquidity_Factor'].median()))
        ), axis=1
    )
    
    return df['Alpha_Factor']

# Example usage:
# df = pd.DataFrame({
#     'date': pd.date_range(start='2023-01-01', periods=100),
#     'open': ...,
#     'high': ...,
#     'low': ...,
#     'close': ...,
#     'amount': ...,
#     'volume': ...
# })
# df.set_index('date', inplace=True)
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
