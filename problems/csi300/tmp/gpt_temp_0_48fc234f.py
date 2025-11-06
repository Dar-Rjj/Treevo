import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Percentile-based momentum acceleration with volume divergence and multiplicative smoothing.
    
    Interpretation:
    - Momentum acceleration measured through percentile ranking across multiple timeframes
    - Volume divergence captures abnormal trading activity relative to recent patterns
    - Multiplicative smoothing enhances signal persistence across regimes
    - Volatility-scaled volume conviction weights signals by market intensity
    - Price range normalization ensures momentum signals are comparable across different price levels
    - Positive values indicate accelerating bullish momentum with volume confirmation
    - Negative values suggest bearish momentum acceleration with volume distribution
    """
    
    # Price range for normalization
    price_range = df['high'] - df['low']
    
    # Momentum components normalized by price range
    intraday_return = (df['close'] - df['open']) / (price_range + 1e-7)
    daily_return = (df['close'] - df['close'].shift(1)) / (price_range + 1e-7)
    
    # Momentum acceleration using percentile ranking
    intraday_momentum_5d = intraday_return.rolling(window=5).apply(lambda x: (x.rank(pct=True).iloc[-1] - 0.5) * 2)
    intraday_momentum_10d = intraday_return.rolling(window=10).apply(lambda x: (x.rank(pct=True).iloc[-1] - 0.5) * 2)
    
    daily_momentum_5d = daily_return.rolling(window=5).apply(lambda x: (x.rank(pct=True).iloc[-1] - 0.5) * 2)
    daily_momentum_10d = daily_return.rolling(window=10).apply(lambda x: (x.rank(pct=True).iloc[-1] - 0.5) * 2)
    
    # Momentum acceleration (change in momentum percentile)
    momentum_accel_5d = intraday_momentum_5d - intraday_momentum_5d.shift(3)
    momentum_accel_10d = intraday_momentum_10d - intraday_momentum_10d.shift(5)
    
    # Volume divergence from recent patterns
    volume_ma_5 = df['volume'].rolling(window=5).mean()
    volume_ma_10 = df['volume'].rolling(window=10).mean()
    volume_divergence = (df['volume'] / (volume_ma_5 + 1e-7) - 1) * (df['volume'] / (volume_ma_10 + 1e-7) - 1)
    
    # Volatility-scaled volume conviction
    range_volatility = price_range.rolling(window=5).std()
    volume_conviction = volume_divergence * (price_range / (range_volatility + 1e-7))
    
    # Multiplicative smoothing across timeframes
    momentum_composite = (
        intraday_momentum_5d * intraday_momentum_10d * 
        np.sign(intraday_momentum_5d * intraday_momentum_10d)
    )
    
    acceleration_composite = (
        momentum_accel_5d * momentum_accel_10d * 
        np.sign(momentum_accel_5d * momentum_accel_10d)
    )
    
    # Regime persistence detection
    momentum_persistence = (
        np.sign(intraday_momentum_5d * intraday_momentum_5d.shift(1)) +
        np.sign(daily_momentum_5d * daily_momentum_5d.shift(1))
    ) / 2
    
    # Combined alpha factor with multiplicative blending
    alpha_factor = (
        momentum_composite * 0.4 +
        acceleration_composite * 0.3 +
        volume_conviction * 0.2 +
        momentum_persistence * 0.1
    )
    
    return alpha_factor
