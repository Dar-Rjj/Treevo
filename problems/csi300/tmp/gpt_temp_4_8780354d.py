import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Momentum Components with Decay
    # Calculate 5-day close momentum
    momentum_5d = df['close'] / df['close'].shift(5) - 1
    
    # Calculate 10-day momentum difference
    momentum_10d = df['close'] / df['close'].shift(10) - 1
    momentum_diff = momentum_5d - momentum_10d
    
    # Apply exponential decay to momentum over 20-day window
    decay_weights = np.exp(-np.arange(20) / 5)  # 5-day decay constant
    decay_weights = decay_weights / decay_weights.sum()
    
    decayed_momentum = momentum_diff.rolling(window=20, min_periods=1).apply(
        lambda x: np.sum(x * decay_weights[-len(x):]), raw=False
    )
    
    # Volume-Weighted Range Efficiency
    # Compute True Range (High, Low, Previous Close)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Close-to-Close return divided by True Range
    close_return = df['close'] / df['close'].shift(1) - 1
    range_efficiency = close_return / true_range.replace(0, np.nan)
    range_efficiency = range_efficiency.fillna(0)
    
    # Volume scaling using current vs 20-day average volume
    vol_avg_20d = df['volume'].rolling(window=20, min_periods=1).mean()
    volume_scaling = df['volume'] / vol_avg_20d.replace(0, np.nan)
    volume_scaling = volume_scaling.fillna(1)
    
    # Asymmetric Signal Enhancement
    # Bullish Asymmetry Condition
    bullish_condition = (df['close'] > df['open']) & (decayed_momentum > 0)
    
    # Bearish Asymmetry Condition
    bearish_condition = (df['close'] < df['open']) & (decayed_momentum < 0)
    
    # Directional multiplier
    directional_multiplier = pd.Series(1.0, index=df.index)
    directional_multiplier[bullish_condition] = 1.5
    directional_multiplier[bearish_condition] = 0.5
    
    # Factor Combination
    # Multiply decayed momentum by range efficiency
    base_factor = decayed_momentum * range_efficiency
    
    # Apply volume weighting
    volume_weighted_factor = base_factor * volume_scaling
    
    # Multiply by directional multiplier based on asymmetry condition
    final_factor = volume_weighted_factor * directional_multiplier
    
    return final_factor
