import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Adaptive Momentum-Volume-Volatility Composite Factor
    
    Economic intuition: Combines momentum, volume, and volatility signals multiplicatively
    to capture stocks with strong price trends supported by volume activity and favorable
    volatility characteristics. The adaptive smoothing and volatility normalization enhance
    signal quality while maintaining interpretability.
    
    Key innovations:
    - Multiplicative combination of momentum, volume, and volatility components
    - Adaptive smoothing using geometric means across asymmetric horizons
    - Volatility normalization for risk adjustment
    - Dollar volume weighting for liquidity consideration
    - Simple, interpretable factor construction
    """
    
    # Momentum component: 5-day and 21-day geometric mean
    mom_5 = df['close'] / df['close'].shift(5) - 1
    mom_21 = df['close'] / df['close'].shift(21) - 1
    momentum = (mom_5 * mom_21).abs() ** 0.5 * np.sign(mom_5 + mom_21)
    
    # Volume component: volume acceleration relative to 10-day average
    vol_ratio = df['volume'] / df['volume'].rolling(window=10).mean()
    volume_signal = vol_ratio - 1
    
    # Volatility component: inverse of normalized daily range
    daily_range = (df['high'] - df['low']) / df['close']
    vol_5 = daily_range.rolling(window=5).mean()
    volatility_signal = 1 / (vol_5 + 1e-7)
    
    # Multiplicative combination
    raw_factor = momentum * volume_signal * volatility_signal
    
    # Adaptive smoothing with geometric mean of 3 and 8-day windows
    smooth_3 = raw_factor.rolling(window=3).apply(lambda x: np.prod(1 + x) ** (1/len(x)) - 1)
    smooth_8 = raw_factor.rolling(window=8).apply(lambda x: np.prod(1 + x) ** (1/len(x)) - 1)
    smoothed_factor = (smooth_3 * smooth_8).abs() ** 0.5 * np.sign(smooth_3 + smooth_8)
    
    # Volatility normalization using 21-day rolling standard deviation
    factor_vol = smoothed_factor.rolling(window=21).std()
    normalized_factor = smoothed_factor / (factor_vol + 1e-7)
    
    # Dollar volume weighting
    dollar_volume = df['close'] * df['volume']
    dollar_weight = dollar_volume / dollar_volume.rolling(window=21).mean()
    
    final_factor = normalized_factor * dollar_weight
    
    return final_factor
