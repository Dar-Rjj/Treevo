import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Return
    df['Daily_Return'] = df['close'].pct_change()
    
    # Identify Volume Spike
    spike_factor = 2
    df['Volume_Change'] = df['volume'] - df['volume'].shift(1)
    df['Volume_Spike'] = (df['Volume_Change'] / df['volume'].shift(1)) > spike_factor
    
    # Calculate Price Change Velocity
    window = 5  # Number of days to consider for the weighted moving average
    weights = list(range(1, window + 1))
    df['Price_Change'] = df['close'] - df['close'].shift(1)
    df['Weighted_Price_Change'] = df['Price_Change'].rolling(window=window).apply(lambda x: sum(w * p for w, p in zip(weights, x)), raw=True) / sum(weights)
    
    # Calculate Volume-Weighted N-day Momentum
    n_days = 5
    df['Volume_Weighted_Momentum'] = (df['Daily_Return'] * df['volume']).rolling(window=n_days).sum()
    df['Adjusted_Volume_Weighted_Momentum'] = df['Volume_Weighted_Momentum'] * (spike_factor if df['Volume_Spike'] else 1)
    
    # Calculate Enhanced Momentum Adjusted Volume
    ema_alpha = 0.2
    df['Volume_Change_Ratio'] = df['volume'] / df['volume'].shift(1)
    df['EMA_Volume'] = df['volume'].ewm(alpha=ema_alpha, adjust=False).mean()
    df['Enhanced_Momentum_Adjusted_Volume'] = df['Volume_Change_Ratio'] * df['EMA_Volume']
    
    # Calculate Price Volatility
    df['High_Low_Range'] = df['high'] - df['low']
    df['Average_High_Low_Range'] = df['High_Low_Range'].rolling(window=n_days).mean()
    
    # Combine Adjusted Momentum and Velocity
    volatility_threshold = 0.5
    volatility_factor = 1.5
    df['Combined_Momentum'] = df['Adjusted_Volume_Weighted_Momentum'] * df['Weighted_Price_Change']
    df['Final_Factor'] = df['Combined_Momentum'] * (volatility_factor if df['Average_High_Low_Range'] > volatility_threshold else 1)
    
    return df['Final_Factor']
