import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Scale Entropy Decay
    df = df.copy()
    
    # Short-Term (5d) Entropy Decay
    vol_ratio_5d_1 = df['volume'] / df['volume'].shift(2)
    range_ratio_5d_1 = (df['high'] - df['low']) / (df['high'].shift(2) - df['low'].shift(2))
    vol_ratio_5d_2 = df['volume'].shift(2) / df['volume'].shift(5)
    range_ratio_5d_2 = (df['high'].shift(2) - df['low'].shift(2)) / (df['high'].shift(5) - df['low'].shift(5))
    entropy_5d = vol_ratio_5d_1 * range_ratio_5d_1 - vol_ratio_5d_2 * range_ratio_5d_2
    
    # Medium-Term (13d) Entropy Decay
    vol_ratio_13d_1 = df['volume'] / df['volume'].shift(5)
    range_ratio_13d_1 = (df['high'] - df['low']) / (df['high'].shift(5) - df['low'].shift(5))
    vol_ratio_13d_2 = df['volume'].shift(5) / df['volume'].shift(13)
    range_ratio_13d_2 = (df['high'].shift(5) - df['low'].shift(5)) / (df['high'].shift(13) - df['low'].shift(13))
    entropy_13d = vol_ratio_13d_1 * range_ratio_13d_1 - vol_ratio_13d_2 * range_ratio_13d_2
    
    # Long-Term (34d) Entropy Decay
    vol_ratio_34d_1 = df['volume'] / df['volume'].shift(13)
    range_ratio_34d_1 = (df['high'] - df['low']) / (df['high'].shift(13) - df['low'].shift(13))
    vol_ratio_34d_2 = df['volume'].shift(13) / df['volume'].shift(34)
    range_ratio_34d_2 = (df['high'].shift(13) - df['low'].shift(13)) / (df['high'].shift(34) - df['low'].shift(34))
    entropy_34d = vol_ratio_34d_1 * range_ratio_34d_1 - vol_ratio_34d_2 * range_ratio_34d_2
    
    # Entropy Pattern Detection
    # Acceleration
    acceleration = (entropy_5d - entropy_13d) * np.sign(entropy_5d)
    
    # Persistence
    entropy_series = entropy_5d
    persistence = pd.Series(0, index=df.index)
    for i in range(4, len(df)):
        if i >= 4:
            window = entropy_series.iloc[i-4:i+1]
            count_up = sum(window.iloc[j] > window.iloc[j-1] for j in range(1, 5))
            count_down = sum(window.iloc[j] < window.iloc[j-1] for j in range(1, 5))
            persistence.iloc[i] = count_up - count_down
    
    # Asymmetry
    asymmetry = pd.Series(0.0, index=df.index)
    mask = (entropy_5d != 0) & (entropy_13d != 0)
    asymmetry[mask] = entropy_5d[mask] / entropy_13d[mask]
    
    # Microstructure Integration
    # Efficiency-Entropy
    efficiency_ratio = (df['close'] - df['open']) / (df['high'] - df['low'])
    efficiency_entropy = efficiency_ratio * entropy_5d
    
    # Volume-Entropy
    volume_density = df['volume'] / (df['high'] - df['low'])
    volume_entropy = volume_density * entropy_5d
    
    # Trade-Size Entropy
    avg_trade_size = df['amount'] / df['volume']
    trade_size_entropy = avg_trade_size * entropy_5d
    
    # Regime Identification
    # Cluster Persistence
    entropy_5d_avg = entropy_5d.rolling(window=5, min_periods=1).mean()
    cluster_persistence = pd.Series(0, index=df.index)
    for i in range(2, len(df)):
        if i >= 2:
            window = entropy_5d.iloc[i-2:i+1]
            avg_window = entropy_5d_avg.iloc[i-2:i+1]
            cluster_persistence.iloc[i] = sum(window.iloc[j] > avg_window.iloc[j] for j in range(len(window)))
    
    # High Regime
    high_regime = entropy_5d > 1.5 * entropy_5d_avg
    
    # Multi-Timeframe Sync
    volume_change = df['volume'] / df['volume'].shift(1) - 1
    multi_timeframe_sync = np.sign(entropy_5d) * np.sign(entropy_13d) * np.sign(volume_change)
    
    # Signal Construction
    # Core Factor
    core_factor = pd.Series(0.0, index=df.index)
    core_factor[high_regime] = volume_entropy[high_regime] * cluster_persistence[high_regime]
    core_factor[~high_regime] = trade_size_entropy[~high_regime] * (1 - cluster_persistence[~high_regime] / 3)
    
    # Decay Enhancement
    decay_enhancement = core_factor * (1 + np.abs(asymmetry))
    
    # Efficiency Weighting
    efficiency_weighting = decay_enhancement * efficiency_entropy
    
    # Alpha Output
    # Confirmation
    confirmation = np.sign(cluster_persistence) * np.sign(multi_timeframe_sync)
    
    # Divergence Adjustment
    divergence_adjustment = 1 + np.abs(entropy_5d - volume_change)
    
    # Final Alpha
    final_alpha = efficiency_weighting * confirmation * divergence_adjustment
    
    return final_alpha
