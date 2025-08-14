import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the daily logarithmic return
    df['log_return'] = np.log(df['close']) - np.log(df['close'].shift(1))
    
    # Adaptive window for momentum based on the 10-day ATR
    atr_10d = (df['high'] - df['low']).rolling(window=10).mean()
    adaptive_window = (atr_10d * 10 / df['close']).round().astype(int).clip(lower=5, upper=20)
    df['momentum_ema_adaptive'] = df['log_return'].rolling(window=adaptive_window).apply(lambda x: x.ewm(span=len(x), adjust=False).mean().iloc[-1], raw=True)
    
    # Calculate the 30-day exponential moving standard deviation of daily logarithmic returns to capture volatility
    df['volatility_ema_30d'] = df['log_return'].ewm(span=30, adjust=False).std()
    
    # Calculate the 5-day exponential moving average (EMA) of volume to capture liquidity
    df['volume_ema_5d'] = df['volume'].ewm(span=5, adjust=False).mean()
    
    # Calculate the 10-day average true range (ATR) for volatility
    df['tr_high_low'] = df['high'] - df['low']
    df['tr_high_close_prev'] = (df['high'] - df['close'].shift()).abs()
    df['tr_low_close_prev'] = (df['low'] - df['close'].shift()).abs()
    df['true_range'] = df[['tr_high_low', 'tr_high_close_prev', 'tr_low_close_prev']].max(axis=1)
    df['atr_10d'] = df['true_range'].rolling(window=10).mean()
    
    # Calculate the 20-day price change to capture longer-term momentum
    df['price_change_20d'] = (df['close'] / df['close'].shift(20)) - 1
    
    # Calculate the 50-day price change to capture even longer-term trends
    df['price_change_50d'] = (df['close'] / df['close'].shift(50)) - 1
    
    # Integrate a simple moving average crossover as a technical indicator
    df['sma_50d'] = df['close'].rolling(window=50).mean()
    df['sma_200d'] = df['close'].rolling(window=200).mean()
    df['sma_crossover'] = df['sma_50d'] > df['sma_200d']
    
    # Incorporate market sentiment using the 10-day EMA of amount
    df['amount_ema_10d'] = df['amount'].ewm(span=10, adjust=False).mean()
    
    # Combine the factors into a single alpha factor
    factor = (
        df['momentum_ema_adaptive'] * df['volatility_ema_30d'] * 
        df['volume_ema_5d'] * df['atr_10d'] * 
        df['price_change_20d'] * df['price_change_50d'] * 
        df['sma_crossover'].astype(int) * df['amount_ema_10d']
    )
    
    # Exponentially weight the final factor
    factor = factor.ewm(span=10, adjust=False).mean()
    
    return factor
