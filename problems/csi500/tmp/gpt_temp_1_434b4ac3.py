import pandas as pd
import pandas as pd

def heuristics_v2(df, short_sma_window=20, long_sma_window=200, vol_window=20, pct_change_window=10, turnover_window=20):
    # Calculate Simple Moving Average (SMA) of Close Prices
    df['SMA_Short'] = df['close'].rolling(window=short_sma_window).mean()
    
    # Compute Volume-Adjusted Volatility
    df['High_Low_Diff'] = df['high'] - df['low']
    df['Volume_Adjusted_Volatility'] = (df['High_Low_Diff'] * df['volume']).rolling(window=vol_window).mean()
    
    # Compute Price Momentum
    df['Price_Momentum'] = (df['close'] - df['SMA_Short']) / df['close'].rolling(window=short_sma_window).mean()
    
    # Incorporate Additional Price Change Metrics
    df['Pct_Change_Close'] = df['close'].pct_change(periods=pct_change_window)
    df['High_Low_Range'] = (df['high'] - df['low']) / df['close']
    
    # Consider Market Trend Alignment
    df['SMA_Long'] = df['close'].rolling(window=long_sma_window).mean()
    df['Trend_Indicator'] = (df['SMA_Short'] > df['SMA_Long']).astype(int)
    
    # Incorporate Liquidity Measures
    df['Daily_Turnover'] = df['volume'] * df['close']
    df['Rolling_Avg_Turnover'] = df['Daily_Turnover'].rolling(window=turnover_window).mean()
    
    # Define the Weights for Each Component
    weights = {
        'Price_Momentum': 0.4,
        'Volume_Adjusted_Volatility': -0.3,
        'Pct_Change_Close': 0.2,
        'High_Low_Range': -0.1
    }
    
    # Ensure the Weights Sum to 1
    assert sum(weights.values()) == 1, "Weights must sum to 1"
    
    # Adjust Weights Based on Market Trend and Liquidity
    def adjust_weights(row):
        trend_weight_adj = 0.5 if row['Trend_Indicator'] == 1 else -0.5
        liquidity_weight_adj = 0.5 if row['Rolling_Avg_Turnover'] > df['Rolling_Avg_Turnover'].median() else -0.5
        return {
            'Price_Momentum': weights['Price_Momentum'] + trend_weight_adj + liquidity_weight_adj,
            'Volume_Adjusted_Volatility': weights['Volume_Adjusted_Volatility'],
            'Pct_Change_Close': weights['Pct_Change_Close'],
            'High_Low_Range': weights['High_Low_Range']
        }
    
    df['Adjusted_Weights'] = df.apply(adjust_weights, axis=1)
    df['Alpha_Factor'] = (
        df['Price_Momentum'] * df['Adjusted_Weights'].apply(lambda x: x['Price_Momentum']) +
        df['Volume_Adjusted_Volatility'] * df['Adjusted_Weights'].apply(lambda x: x['Volume_Adjusted_Volatility']) +
        df['Pct_Change_Close'] * df['Adjusted_Weights'].apply(lambda x: x['Pct_Change_Close']) +
        df['High_Low_Range'] * df['Adjusted_Weights'].apply(lambda x: x['High_Low_Range'])
    )
    
    return df['Alpha_Factor']
