import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price-Volume Divergence Momentum
    # Volatility-Normalized Momentum
    df['mom_5d'] = (df['close'] / df['close'].shift(5) - 1) / df['high'].rolling(6).apply(lambda x: (x.max() - x.min()), raw=True).shift(-5)
    df['mom_10d'] = (df['close'] / df['close'].shift(10) - 1) / df['high'].rolling(11).apply(lambda x: (x.max() - x.min()), raw=True).shift(-10)
    
    # Volume Persistence Regime
    df['volume_trend_strength'] = np.sign(df['volume'].diff()) * np.sign(df['volume'].diff().shift(1))
    
    def count_same_direction(arr):
        count = 0
        for i in range(1, len(arr)):
            if np.sign(arr[i]) == np.sign(arr[i-1]) and arr[i] != 0:
                count += 1
        return count
    
    df['volume_persistence'] = df['volume'].diff().rolling(6).apply(count_same_direction, raw=False)
    
    # Divergence Blend
    df['bullish_div'] = df['mom_5d'] + (1 - df['volume_persistence'] / 5)
    df['bearish_div'] = df['mom_10d'] + (df['volume_persistence'] / 5)
    
    # Range Efficiency Momentum
    # Multi-timeframe Efficiency
    df['eff_3d'] = abs(df['close'] - df['close'].shift(3)) / df['high'].rolling(4).apply(lambda x: sum(x.max() - x.min() for _ in range(3)), raw=False).shift(-3)
    df['eff_5d'] = abs(df['close'] - df['close'].shift(5)) / df['high'].rolling(6).apply(lambda x: sum(x.max() - x.min() for _ in range(5)), raw=False).shift(-5)
    
    # Efficiency Trend
    df['eff_mom'] = df['eff_3d'] / df['eff_5d']
    
    def count_efficiency_increase(arr):
        count = 0
        for i in range(1, len(arr)):
            if arr[i] > arr[i-1]:
                count += 1
        return count
    
    df['eff_persistence'] = df['eff_3d'].rolling(6).apply(count_efficiency_increase, raw=False)
    
    # Volume-Confirmed Extreme Reversal
    # Volatility-Adjusted Extremes
    vol_5d = df['high'].rolling(6).apply(lambda x: (x.max() - x.min()), raw=True).shift(-5)
    df['norm_move'] = df['close'].diff() / vol_5d
    df['volume_confirmation'] = (df['volume'] / df['volume'].shift(1) > 2).astype(int)
    
    # Reversal Detection
    df['extreme_move'] = (abs(df['norm_move']) > 2).astype(int)
    df['opposite_move'] = (np.sign(df['close'].diff()) != np.sign(df['close'].diff().shift(-1))).astype(int)
    df['volume_reversal'] = df['volume_confirmation'] & df['opposite_move']
    
    # Amount Flow Regime Detection
    # Directional Flow Strength
    up_days = (df['close'] > df['close'].shift(1)).rolling(5).sum()
    down_days = (df['close'] < df['close'].shift(1)).rolling(5).sum()
    df['net_flow_ratio'] = (up_days - down_days) / (up_days + down_days)
    
    def consecutive_flow_direction(arr):
        max_consecutive = 0
        current = 0
        for i in range(1, len(arr)):
            if np.sign(arr[i]) == np.sign(arr[i-1]):
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0
        return max_consecutive
    
    df['flow_persistence'] = df['net_flow_ratio'].rolling(6).apply(consecutive_flow_direction, raw=False)
    
    # Flow Momentum
    df['flow_accel'] = df['net_flow_ratio'] / df['net_flow_ratio'].shift(3)
    df['flow_stability'] = df['net_flow_ratio'].rolling(6).std()
    
    # Volatility-Volume Regime Alignment
    # Regime Classification
    df['high_vol'] = (df['high'].rolling(6).apply(lambda x: (x.max() - x.min()), raw=True).shift(-5) / df['close'].shift(5) > 0.02).astype(int)
    df['volume_regime'] = (df['volume'] / df['volume'].shift(5) > 1.2).astype(int)
    
    # Combine factors with weights
    factor = (
        0.25 * df['bullish_div'] + 
        0.20 * df['eff_mom'] + 
        0.15 * df['volume_reversal'] + 
        0.20 * df['net_flow_ratio'] + 
        0.20 * (df['high_vol'] * df['volume_regime'])
    )
    
    return factor
