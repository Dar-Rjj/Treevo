import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price-Volume Asymmetry Analysis
    # Short-term divergence: (Close - Close[3]) / (Sum(Volume[t-2:t]) / 3)
    price_change_3d = df['close'] - df['close'].shift(3)
    volume_avg_3d = (df['volume'].shift(2) + df['volume'].shift(1) + df['volume']) / 3
    price_volume_divergence = price_change_3d / volume_avg_3d
    
    # Directional volume bias: (Sum(Volume when Close > Close[1]) - Sum(Volume when Close < Close[1])) / Sum(Volume[t-4:t])
    up_days = df['close'] > df['close'].shift(1)
    down_days = df['close'] < df['close'].shift(1)
    
    volume_up_5d = pd.Series(np.zeros(len(df)), index=df.index)
    volume_down_5d = pd.Series(np.zeros(len(df)), index=df.index)
    
    for i in range(4, len(df)):
        window_up = up_days.iloc[i-4:i+1]
        window_down = down_days.iloc[i-4:i+1]
        volume_up_5d.iloc[i] = df['volume'].iloc[i-4:i+1][window_up].sum()
        volume_down_5d.iloc[i] = df['volume'].iloc[i-4:i+1][window_down].sum()
    
    total_volume_5d = df['volume'].rolling(window=5, min_periods=5).sum()
    directional_volume_bias = (volume_up_5d - volume_down_5d) / total_volume_5d
    
    # Asymmetry ratio: price-volume_divergence / directional_volume_bias
    asymmetry_ratio = price_volume_divergence / directional_volume_bias.replace(0, np.nan)
    
    # Regime Classification
    bullish_regime = (asymmetry_ratio > 0.8) & (df['close'] > df['close'].shift(5))
    bearish_regime = (asymmetry_ratio < -0.8) & (df['close'] < df['close'].shift(5))
    neutral_regime = (~bullish_regime) & (~bearish_regime) & (asymmetry_ratio.abs() <= 0.8)
    
    # Regime-Specific Momentum
    # Bullish persistence: Count(Close > Close[1] over 5 days) / 5
    bullish_persistence = pd.Series(np.zeros(len(df)), index=df.index)
    for i in range(4, len(df)):
        window = df['close'].iloc[i-4:i+1]
        bullish_persistence.iloc[i] = (window > window.shift(1)).sum() / 5
    
    # Bearish decay: Count(Close < Close[1] over 5 days) / 5
    bearish_decay = pd.Series(np.zeros(len(df)), index=df.index)
    for i in range(4, len(df)):
        window = df['close'].iloc[i-4:i+1]
        bearish_decay.iloc[i] = (window < window.shift(1)).sum() / 5
    
    # Volume confirmation: (Volume when regime_consistent) / Volume[1]
    volume_confirmation = pd.Series(np.ones(len(df)), index=df.index)
    for i in range(1, len(df)):
        if bullish_regime.iloc[i] and df['close'].iloc[i] > df['close'].iloc[i-1]:
            volume_confirmation.iloc[i] = df['volume'].iloc[i] / df['volume'].iloc[i-1]
        elif bearish_regime.iloc[i] and df['close'].iloc[i] < df['close'].iloc[i-1]:
            volume_confirmation.iloc[i] = df['volume'].iloc[i] / df['volume'].iloc[i-1]
    
    # Range Dynamics
    # Short-term efficiency: (Close - Lowest(Low,3)) / (Highest(High,3) - Lowest(Low,3))
    low_3d = df['low'].rolling(window=3, min_periods=3).min()
    high_3d = df['high'].rolling(window=3, min_periods=3).max()
    short_term_efficiency = (df['close'] - low_3d) / (high_3d - low_3d).replace(0, np.nan)
    
    # Medium-term utilization: (Close - Lowest(Low,8)) / (Highest(High,8) - Lowest(Low,8))
    low_8d = df['low'].rolling(window=8, min_periods=8).min()
    high_8d = df['high'].rolling(window=8, min_periods=8).max()
    medium_term_utilization = (df['close'] - low_8d) / (high_8d - low_8d).replace(0, np.nan)
    
    # Range divergence: short-term_efficiency - medium-term_utilization
    range_divergence = short_term_efficiency - medium_term_utilization
    
    # Composite Factor
    factor = pd.Series(np.zeros(len(df)), index=df.index)
    
    # Bullish: asymmetry_ratio * momentum_persistence * range_efficiency
    bullish_factor = asymmetry_ratio * bullish_persistence * short_term_efficiency * volume_confirmation
    factor[bullish_regime] = bullish_factor[bullish_regime]
    
    # Bearish: asymmetry_ratio * momentum_decay * range_utilization
    bearish_factor = asymmetry_ratio * bearish_decay * medium_term_utilization * volume_confirmation
    factor[bearish_regime] = bearish_factor[bearish_regime]
    
    # Neutral: asymmetry_ratio * range_divergence
    neutral_factor = asymmetry_ratio * range_divergence
    factor[neutral_regime] = neutral_factor[neutral_regime]
    
    return factor
