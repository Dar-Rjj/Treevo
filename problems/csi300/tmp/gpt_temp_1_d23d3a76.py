import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the daily return
    df['daily_return'] = df['close'].pct_change()
    
    # Calculate the 10-day average of daily returns to capture short-term momentum
    df['momentum_10d'] = df['daily_return'].ewm(span=10, adjust=False).mean()
    
    # Calculate the 30-day standard deviation of daily returns to capture volatility
    df['volatility_30d'] = df['daily_return'].rolling(window=30).std()
    
    # Calculate the 5-day exponential moving average (EMA) of volume to capture liquidity
    df['volume_ema_5d'] = df['volume'].ewm(span=5, adjust=False).mean()
    
    # Calculate the 10-day average true range (ATR) for volatility
    df['tr_high_low'] = df['high'] - df['low']
    df['tr_high_close_prev'] = (df['high'] - df['close'].shift()).abs()
    df['tr_low_close_prev'] = (df['low'] - df['close'].shift()).abs()
    df['true_range'] = df[['tr_high_low', 'tr_high_close_prev', 'tr_low_close_prev']].max(axis=1)
    df['atr_10d'] = df['true_range'].ewm(span=10, adjust=False).mean()
    
    # Calculate the 20-day price change to capture longer-term momentum
    df['price_change_20d'] = (df['close'] / df['close'].shift(20)) - 1
    
    # Calculate the 50-day price change to capture even longer-term trends
    df['price_change_50d'] = (df['close'] / df['close'].shift(50)) - 1
    
    # Integrate a simple moving average crossover as a technical indicator
    df['sma_50d'] = df['close'].ewm(span=50, adjust=False).mean()
    df['sma_200d'] = df['close'].ewm(span=200, adjust=False).mean()
    df['sma_crossover'] = df['sma_50d'] > df['sma_200d']
    
    # Combine the factors into a single alpha factor
    factor = (
        df['momentum_10d'] * df['volatility_30d'] * 
        df['volume_ema_5d'] * df['atr_10d'] * 
        df['price_change_20d'] * df['price_change_50d'] * 
        df['sma_crossover'].astype(int)
    )
    
    # Add a seasonality factor (e.g., month of the year)
    df['month'] = df.index.month
    df['seasonality_factor'] = df['month'].map({1: 0.8, 2: 0.9, 3: 1.0, 4: 1.1, 5: 1.2, 6: 1.3, 7: 1.4, 8: 1.5, 9: 1.6, 10: 1.7, 11: 1.8, 12: 1.9})
    factor *= df['seasonality_factor']
    
    # Avoid division by zero and handle NaN values
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return factor
