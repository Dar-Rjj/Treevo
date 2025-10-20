import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Efficiency-Momentum with Volume-Range Confirmation factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Price Efficiency Ratio components
    for window in [5, 20, 60]:
        # Calculate returns
        returns = df['close'].pct_change(window)
        # Calculate absolute returns sum
        abs_returns_sum = df['close'].pct_change().abs().rolling(window=window, min_periods=1).sum()
        # Efficiency ratio
        efficiency = returns / abs_returns_sum.replace(0, np.nan)
        efficiency = efficiency.fillna(0)
        
        if window == 5:
            eff_short = efficiency
        elif window == 20:
            eff_medium = efficiency
        else:
            eff_long = efficiency
    
    # Volume-Weighted Efficiency
    vol_accel = df['volume'].rolling(window=5, min_periods=1).mean() / df['volume'].rolling(window=20, min_periods=1).mean()
    vol_accel = vol_accel.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Efficiency-volume correlation
    eff_vol_corr = np.sign(eff_short) * vol_accel
    
    # Efficiency persistence (short-term)
    eff_sign = np.sign(eff_short)
    eff_persistence = pd.Series(index=df.index, dtype=int)
    for i in range(1, len(df)):
        if eff_sign.iloc[i] == eff_sign.iloc[i-1] and eff_sign.iloc[i] != 0:
            eff_persistence.iloc[i] = eff_persistence.iloc[i-1] + 1
        else:
            eff_persistence.iloc[i] = 0
    eff_persistence = eff_persistence.fillna(0)
    
    # Regime-Dependent Momentum
    vol_20 = df['close'].pct_change().rolling(window=20, min_periods=1).std()
    vol_60 = df['close'].pct_change().rolling(window=60, min_periods=1).std()
    vol_ratio = vol_20 / vol_60.replace(0, np.nan)
    vol_ratio = vol_ratio.fillna(1)
    
    # High vol momentum
    ret_5 = df['close'].pct_change(5)
    high_vol_momentum = ret_5 * vol_ratio
    
    # Low vol momentum with range compression
    ret_20 = df['close'].pct_change(20)
    range_5_avg = (df['high'] - df['low']).rolling(window=5, min_periods=1).mean()
    range_20_avg = (df['high'] - df['low']).rolling(window=20, min_periods=1).mean()
    range_compression = range_5_avg / range_20_avg.replace(0, np.nan)
    range_compression = range_compression.fillna(1)
    low_vol_momentum = ret_20 * range_compression
    
    # Momentum acceleration
    momentum_accel = high_vol_momentum - high_vol_momentum.shift(1)
    momentum_accel = momentum_accel.fillna(0)
    
    # Liquidity Absorption
    # Volume-price divergence
    price_dir = np.sign(df['close'] - df['open'])
    vol_dir = np.sign(df['volume'] - df['volume'].shift(1))
    vol_dir = vol_dir.fillna(0)
    vol_price_div = price_dir * vol_dir
    
    # Absorption ratio (simplified as price-volume relationship)
    absorption_ratio = (df['close'] - df['open']) / df['volume'].replace(0, np.nan)
    absorption_ratio = absorption_ratio.fillna(0)
    
    # Flow persistence
    flow_dir = np.sign(absorption_ratio)
    flow_persistence = pd.Series(index=df.index, dtype=int)
    for i in range(1, len(df)):
        if flow_dir.iloc[i] == flow_dir.iloc[i-1] and flow_dir.iloc[i] != 0:
            flow_persistence.iloc[i] = flow_persistence.iloc[i-1] + 1
        else:
            flow_persistence.iloc[i] = 0
    flow_persistence = flow_persistence.fillna(0)
    
    # Range Dynamics
    # Range efficiency
    daily_range = df['high'] - df['low']
    range_efficiency = (df['close'] - df['open']) / daily_range.replace(0, np.nan)
    range_efficiency = range_efficiency.fillna(0)
    
    # Range expansion
    range_expansion = (daily_range > range_5_avg).astype(float)
    
    # Compression timing
    compression_timing = range_5_avg / range_20_avg.replace(0, np.nan)
    compression_timing = compression_timing.fillna(1)
    
    # Signal Convergence
    # Efficiency-momentum alignment
    eff_momentum_align = eff_short * high_vol_momentum
    
    # Volume-flow confirmation
    vol_flow_conf = vol_accel * np.sign(eff_short)
    
    # Multi-timeframe scoring
    timeframe_alignment = (np.sign(eff_short) + np.sign(eff_medium) + np.sign(eff_long)) / 3
    
    # Combine all components with weights
    factor = (
        0.15 * eff_short +
        0.12 * eff_medium +
        0.10 * eff_long +
        0.08 * eff_vol_corr +
        0.06 * eff_persistence +
        0.10 * high_vol_momentum +
        0.08 * low_vol_momentum +
        0.05 * momentum_accel +
        0.04 * vol_price_div +
        0.04 * absorption_ratio +
        0.03 * flow_persistence +
        0.05 * range_efficiency +
        0.03 * range_expansion +
        0.03 * compression_timing +
        0.04 * eff_momentum_align +
        0.03 * vol_flow_conf +
        0.02 * timeframe_alignment
    )
    
    return factor.fillna(0)
