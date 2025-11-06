import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Volatility-normalized momentum with volume trend alignment
    # Combines mean reversion with range efficiency
    
    # Calculate Average True Range (ATR) for volatility normalization
    high_low = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift(1)).abs()
    low_close_prev = (df['low'] - df['close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = true_range.rolling(window=5).mean()
    
    # Volatility-normalized momentum (5-day momentum scaled by ATR)
    price_momentum = (df['close'] - df['close'].shift(5)) / atr
    
    # Volume trend alignment (current volume vs 5-day volume trend)
    volume_trend = df['volume'].rolling(window=5).apply(lambda x: (x[-1] - x.mean()) / x.std() if x.std() > 0 else 0)
    
    # Range efficiency (current day's price range efficiency)
    daily_range = df['high'] - df['low']
    range_efficiency = (df['close'] - df['open']) / daily_range.replace(0, 1)
    
    # Mean reversion component (deviation from 10-day moving average)
    ma_10 = df['close'].rolling(window=10).mean()
    mean_reversion = (df['close'] - ma_10) / atr
    
    # Combine components: momentum aligned with volume trend, adjusted by range efficiency and mean reversion
    alpha_factor = price_momentum * volume_trend * range_efficiency * mean_reversion
    
    return alpha_factor
