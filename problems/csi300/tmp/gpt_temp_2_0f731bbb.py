import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate true range
    df['prev_close'] = df['close'].shift(1)
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['prev_close']),
            abs(df['low'] - df['prev_close'])
        )
    )
    
    # Calculate 5-day absolute price movement
    df['abs_movement_5d'] = abs(df['close'] - df['close'].shift(5))
    
    # Calculate range efficiency ratio
    df['range_efficiency'] = df['abs_movement_5d'] / (df['true_range'].rolling(window=5, min_periods=1).sum())
    df['range_efficiency'] = df['range_efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Calculate mid-range price
    df['mid_range'] = (df['high'] + df['low']) / 2
    
    # Compute 5-day momentum
    df['momentum_5d'] = df['mid_range'] - df['mid_range'].shift(5)
    
    # Calculate momentum acceleration
    df['momentum_accel'] = df['momentum_5d'] - df['momentum_5d'].shift(1)
    
    # Calculate volume trend
    df['volume_trend'] = (df['volume'] - df['volume'].shift(5)) / df['volume'].shift(5).replace(0, np.nan)
    df['volume_trend'] = df['volume_trend'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Calculate volume percentile rank (20-day)
    df['volume_rank'] = df['volume'].rolling(window=20, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Calculate volume-efficiency correlation (10-day)
    df['vol_eff_corr'] = df['volume'].rolling(window=10, min_periods=1).corr(df['range_efficiency'])
    
    # Generate base momentum signal
    momentum_signal = np.sign(df['momentum_5d']) * np.sqrt(abs(df['momentum_5d']))
    
    # Volume-weighted momentum confirmation
    volume_confirmation = np.where(
        (df['momentum_5d'] > 0) & (df['volume_trend'] < 0), -1,  # Positive momentum, negative volume → reversal
        np.where(
            (df['momentum_5d'] < 0) & (df['volume_trend'] > 0), 1,  # Negative momentum, positive volume → reversal
            np.sign(df['momentum_5d'])  # Aligned momentum & volume → continuation
        )
    )
    
    # Efficiency-regime adaptive positioning
    efficiency_threshold = df['range_efficiency'].rolling(window=20, min_periods=1).quantile(0.7)
    
    # High efficiency: momentum following, Low efficiency: mean reversion bias
    regime_signal = np.where(
        df['range_efficiency'] > efficiency_threshold,
        momentum_signal,  # High efficiency - follow momentum
        -momentum_signal * 0.5  # Low efficiency - mean reversion bias with reduced strength
    )
    
    # Dynamic signal strength based on efficiency level
    efficiency_weight = np.minimum(df['range_efficiency'] * 2, 1.0)
    
    # Combine signals with volume confirmation and efficiency weighting
    final_signal = (
        regime_signal * 
        volume_confirmation * 
        efficiency_weight * 
        (1 + df['momentum_accel'] * 0.1)
    )
    
    # Apply volume rank as additional filter
    volume_filter = np.where(df['volume_rank'] > 0.3, 1, 0.5)
    final_signal = final_signal * volume_filter
    
    # Normalize and return
    result = pd.Series(final_signal, index=df.index)
    result = (result - result.rolling(window=20, min_periods=1).mean()) / result.rolling(window=20, min_periods=1).std()
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return result
