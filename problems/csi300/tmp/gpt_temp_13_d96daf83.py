import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Advanced alpha factor combining volatility-normalized momentum with volume acceleration
    and breakout detection. This factor identifies stocks with strong momentum signals that
    are confirmed by volume acceleration and exhibit breakout characteristics from recent
    trading ranges, suggesting high-probability continuation patterns.
    """
    # Volatility-normalized momentum (10-day price change normalized by 15-day ATR)
    atr = (
        (df['high'] - df['low']).rolling(15).mean() + 
        abs(df['high'] - df['close'].shift(1)).rolling(15).mean() + 
        abs(df['low'] - df['close'].shift(1)).rolling(15).mean()
    ) / 3
    momentum = (df['close'] - df['close'].shift(10)) / (atr + 1e-7)
    
    # Volume acceleration with momentum alignment (volume trend confirming price trend)
    volume_trend = df['volume'].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    price_trend = df['close'].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    volume_acceleration = volume_trend * np.sign(price_trend) / (df['volume'].rolling(10).std() + 1e-7)
    
    # Breakout detection from recent consolidation (price breaking out of 20-day range)
    recent_high = df['high'].rolling(20).max()
    recent_low = df['low'].rolling(20).min()
    breakout_strength = (df['close'] - recent_low) / (recent_high - recent_low + 1e-7)
    
    # Volatility compression preceding breakout (low volatility before potential move)
    volatility_compression = (df['high'] - df['low']).rolling(10).std() / (atr + 1e-7)
    
    # Multiplicative combination emphasizing momentum-volume-breakout synergy
    factor = momentum * volume_acceleration * breakout_strength * (1 / (volatility_compression + 1e-7))
    
    return factor
