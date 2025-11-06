import pandas as pd
import pandas as pd

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Dynamic regime-switching momentum factor with adaptive volume confirmation
    and price-level based signal enhancement.
    
    Factor Logic:
    1. Multi-timeframe momentum with regime-dependent weighting
    2. Dynamic volatility regime detection using rolling percentiles
    3. Volume momentum acceleration with asymmetric scaling
    4. Price-level based signal amplification at support/resistance zones
    5. Adaptive signal dampening in high-noise periods
    
    Interpretation:
    - Positive values: Bullish momentum with volume acceleration
    - Negative values: Bearish momentum with volume deceleration
    - Higher magnitude: Strong signals at key price levels with volume confirmation
    - Adaptive to market volatility and volume conditions
    """
    
    # Multi-timeframe momentum calculation
    mom_2 = (df['close'] - df['close'].shift(2)) / df['close'].shift(2)
    mom_4 = (df['close'] - df['close'].shift(4)) / df['close'].shift(4)
    mom_6 = (df['close'] - df['close'].shift(6)) / df['close'].shift(6)
    
    # Dynamic volatility regime detection
    true_range = df['high'] - df['low']
    vol_percentile = true_range.rolling(window=15).apply(lambda x: (x.iloc[-1] > x.quantile(0.7)) * 1 + (x.iloc[-1] < x.quantile(0.3)) * -1, raw=False)
    
    # Regime-dependent momentum blending
    momentum_blend = pd.Series(index=df.index, dtype=float)
    
    # High volatility regime (vol_percentile = 1): Emphasize medium-term momentum
    high_vol_mask = vol_percentile == 1
    momentum_blend[high_vol_mask] = (
        mom_2[high_vol_mask] * 0.2 + 
        mom_4[high_vol_mask] * 0.5 +
        mom_6[high_vol_mask] * 0.3
    )
    
    # Low volatility regime (vol_percentile = -1): Emphasize short-term momentum
    low_vol_mask = vol_percentile == -1
    momentum_blend[low_vol_mask] = (
        mom_2[low_vol_mask] * 0.5 + 
        mom_4[low_vol_mask] * 0.3 +
        mom_6[low_vol_mask] * 0.2
    )
    
    # Normal volatility regime (vol_percentile = 0): Balanced approach
    normal_vol_mask = ~high_vol_mask & ~low_vol_mask
    momentum_blend[normal_vol_mask] = (
        mom_2[normal_vol_mask] * 0.4 + 
        mom_4[normal_vol_mask] * 0.4 +
        mom_6[normal_vol_mask] * 0.2
    )
    
    # Volume acceleration with dynamic thresholds
    vol_change = (df['volume'] - df['volume'].shift(3)) / (df['volume'].shift(3) + 1e-7)
    vol_accel = vol_change - vol_change.shift(2)
    
    # Dynamic volume scaling based on acceleration magnitude
    vol_scale = pd.Series(1.0, index=df.index)
    strong_accel_up = vol_accel > vol_accel.rolling(window=10).quantile(0.75)
    strong_accel_down = vol_accel < vol_accel.rolling(window=10).quantile(0.25)
    vol_scale[strong_accel_up] = 1.4
    vol_scale[strong_accel_down] = 0.7
    
    # Price-level based signal enhancement
    rolling_low = df['low'].rolling(window=10).min()
    rolling_high = df['high'].rolling(window=10).max()
    current_mid = (df['high'] + df['low']) / 2
    
    # Support and resistance zone detection
    near_support = (current_mid - rolling_low) / (rolling_high - rolling_low + 1e-7) < 0.3
    near_resistance = (current_mid - rolling_low) / (rolling_high - rolling_low + 1e-7) > 0.7
    
    # Core factor construction
    core_factor = momentum_blend * vol_scale
    
    # Price-level signal enhancement
    factor = core_factor.copy()
    
    # Amplify bullish signals near support
    bullish_support = near_support & (momentum_blend > 0)
    factor[bullish_support] = core_factor[bullish_support] * 1.6
    
    # Amplify bearish signals near resistance
    bearish_resistance = near_resistance & (momentum_blend < 0)
    factor[bearish_resistance] = core_factor[bearish_resistance] * 1.6
    
    # Dampen signals in high volatility with low volume acceleration
    high_vol_low_accel = high_vol_mask & (vol_accel < 0)
    factor[high_vol_low_accel] = factor[high_vol_low_accel] * 0.6
    
    # Volume breakout confirmation bonus
    volume_breakout = (df['volume'] > df['volume'].rolling(window=15).quantile(0.8)) & (vol_accel > 0)
    factor[volume_breakout] = factor[volume_breakout] * 1.3
    
    return factor
