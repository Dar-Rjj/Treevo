import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Return
    df['daily_return'] = df['close'].pct_change()
    
    # Compute Exponential Moving Average (EMA) of Returns
    ema_window = 10
    df['ema_return'] = df['daily_return'].ewm(span=ema_window, adjust=False).mean()
    
    # Identify High and Low Volatility Days
    def true_range(row):
        tr1 = row['high'] - row['low']
        tr2 = abs(row['high'] - row['close'].shift(1))
        tr3 = abs(row['low'] - row['close'].shift(1))
        return max(tr1, tr2, tr3)
    
    df['true_range'] = df.apply(true_range, axis=1)
    atr_window = 14
    df['atr'] = df['true_range'].rolling(window=atr_window).mean()
    
    volatility_threshold = df['atr'].quantile(0.75)  # Set the threshold to the 75th percentile of ATR
    df['high_volatility'] = df['atr'] > volatility_threshold
    df['low_volatility'] = df['atr'] <= volatility_threshold
    
    # Filter Days by Volatility
    high_vol_days = df[df['high_volatility']]
    low_vol_days = df[df['low_volatility']]
    
    # Calculate Volume Weighted Momentum
    df['vol_weighted_momentum'] = df['daily_return'] * df['volume']
    
    # Compute Momentum Difference
    high_vol_momentum = high_vol_days['vol_weighted_momentum'].sum()
    low_vol_momentum = low_vol_days['vol_weighted_momentum'].sum()
    momentum_difference = high_vol_momentum - low_vol_momentum
    
    # Perform Sentiment Analysis
    # Assume a column 'sentiment_score' is available in the dataframe which represents the sentiment score for each day
    if 'sentiment_score' not in df.columns:
        raise ValueError("Sentiment scores are required but not found in the DataFrame.")
    
    # Adjust for Overall Market Trend
    # Assume a column 'market_index_close' is available in the dataframe which represents the market index close price
    if 'market_index_close' not in df.columns:
        raise ValueError("Market index close prices are required but not found in the DataFrame.")
    
    df['market_return'] = df['market_index_close'].pct_change()
    df['adjusted_ema_return'] = df['ema_return'] - df['market_return']
    
    # Generate Final Alpha Factor
    df['alpha_factor'] = (momentum_difference + df['sentiment_score']) * df['adjusted_ema_return']
    
    return df['alpha_factor']
