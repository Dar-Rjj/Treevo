import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the daily logarithmic return
    df['log_return'] = np.log(df['close']).diff()
    
    # Calculate the 10-day EMA of log returns to capture short-term momentum
    df['momentum_ema_10d'] = df['log_return'].ewm(span=10, adjust=False).mean()
    
    # Calculate the 30-day EMA of log returns for volatility
    df['volatility_ema_30d'] = df['log_return'].ewm(span=30, adjust=False).std()
    
    # Calculate the 5-day EMA of volume to capture liquidity
    df['volume_ema_5d'] = df['volume'].ewm(span=5, adjust=False).mean()
    
    # Calculate the 10-day ATR for volatility
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
    df['sma_50d'] = df['close'].rolling(window=50).mean()
    df['sma_200d'] = df['close'].rolling(window=200).mean()
    df['sma_crossover'] = (df['sma_50d'] > df['sma_200d']).astype(int)
    
    # Calculate the RSI
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    
    # Combine the factors into a single alpha factor with multiplicative interactions
    factor = (
        df['momentum_ema_10d'] * df['volatility_ema_30d'] * 
        df['volume_ema_5d'] * df['atr_10d'] * 
        df['price_change_20d'] * df['price_change_50d'] * 
        df['sma_crossover'] * df['rsi']
    )
    
    return factor
