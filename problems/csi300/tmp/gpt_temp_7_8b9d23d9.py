import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Simple momentum-volume alignment with extreme volume detection.
    
    Factor Logic:
    1. Short-term momentum (5-day) normalized by price range volatility
    2. Volume acceleration relative to recent average
    3. Extreme volume detection for potential reversal signals
    4. Combined momentum-volume alignment with reversal adjustments
    
    Interpretation:
    - Positive values indicate bullish momentum with volume confirmation
    - Negative values indicate bearish momentum with volume confirmation
    - Extreme volume with counter-trend momentum suggests potential reversals
    - Simple, interpretable combination of price and volume dynamics
    """
    
    # 5-day momentum normalized by 5-day price range volatility
    momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    price_range_vol = (df['high'] - df['low']).rolling(window=5).std()
    norm_momentum = momentum_5d / (price_range_vol + 1e-7)
    
    # Volume acceleration (current volume vs 5-day average)
    volume_accel = df['volume'] / df['volume'].rolling(window=5).mean()
    
    # Extreme volume detection using 10-day rolling percentile
    volume_percentile = df['volume'].rolling(window=10).apply(lambda x: (x[-1] > np.percentile(x, 80)))
    
    # Core factor: momentum amplified by volume acceleration
    factor = norm_momentum * volume_accel
    
    # Apply reversal logic for extreme volume scenarios
    extreme_bullish_reversal = (volume_percentile == 1) & (momentum_5d < 0)
    extreme_bearish_reversal = (volume_percentile == 1) & (momentum_5d > 0)
    
    # Enhance factor for potential bullish reversals, reduce for bearish reversals
    factor = factor.mask(extreme_bullish_reversal, factor * 2.0)
    factor = factor.mask(extreme_bearish_reversal, factor * 0.5)
    
    return factor
