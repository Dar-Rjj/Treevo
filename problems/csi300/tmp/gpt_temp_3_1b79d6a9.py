import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Volatility Regime Shift
    intraday_vol = (data['high'] - data['low']) / data['open']
    vol_median_10d = intraday_vol.rolling(window=10, min_periods=5).median()
    regime_shift_ratio = intraday_vol / vol_median_10d
    
    short_momentum = data['close'] / data['close'].shift(3)
    medium_momentum = data['close'] / data['close'].shift(8)
    momentum_diff = short_momentum - medium_momentum
    
    factor1 = regime_shift_ratio * momentum_diff
    
    # Volume-Intensity Price Pressure
    intensity = (data['amount'] / data['volume']) * data['volume']
    intensity_mean_5d = intensity.rolling(window=5, min_periods=3).mean()
    intensity_change = intensity / intensity_mean_5d
    
    pressure_direction = (data['close'] - data['open']) / data['open']
    
    factor2 = intensity_change * pressure_direction
    
    # Range Expansion Momentum Divergence
    current_range = (data['high'] - data['low']) / data['close']
    prev_range = (data['high'].shift(1) - data['low'].shift(1)) / data['close'].shift(1)
    expansion_ratio = current_range / prev_range
    
    fast_momentum = data['close'] / data['close'].shift(2)
    slow_momentum = data['close'] / data['close'].shift(6)
    momentum_divergence = fast_momentum - slow_momentum
    
    factor3 = expansion_ratio * momentum_divergence
    
    # Amount-Based Liquidity Signal
    liquidity = (data['amount'] / data['volume']) / data['close']
    liquidity_median_8d = liquidity.rolling(window=8, min_periods=5).median()
    liquidity_change = liquidity / liquidity_median_8d
    
    short_reversal = data['close'] / data['close'].shift(1) - 1
    medium_trend = data['close'] / data['close'].shift(5)
    reversal_strength = short_reversal / medium_trend
    
    factor4 = liquidity_change * reversal_strength
    
    # Open-Close Efficiency Ratio
    efficiency = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    efficiency_mean_12d = efficiency.rolling(window=12, min_periods=8).mean()
    efficiency_deviation = efficiency / efficiency_mean_12d
    
    volume_momentum = data['volume'] / data['volume'].shift(1)
    volume_consistency = data['volume'] / data['volume'].shift(3)
    combined_volume = volume_momentum * volume_consistency
    
    factor5 = efficiency_deviation * combined_volume
    
    # Volatility-Weighted Return Persistence
    daily_return = data['close'] / data['close'].shift(1) - 1
    daily_vol = (data['high'] - data['low']) / data['close']
    vol_adjusted_return = daily_return / daily_vol
    
    # Calculate persistence score (sum of same-sign returns over 3 days)
    sign_returns = np.sign(vol_adjusted_return)
    persistence_score = vol_adjusted_return.rolling(window=3, min_periods=2).apply(
        lambda x: x[x * x.iloc[-1] > 0].sum() if len(x) > 1 else 0, raw=False
    )
    
    vol_regime_multiplier = daily_vol / daily_vol.rolling(window=10, min_periods=6).median()
    
    factor6 = persistence_score * vol_regime_multiplier
    
    # Volume-Cluster Price Breakout
    volume_zscore = (data['volume'] - data['volume'].rolling(window=15, min_periods=10).mean()) / data['volume'].rolling(window=15, min_periods=10).std()
    
    # Calculate cluster intensity (consecutive days with z-score > 1)
    high_volume = (volume_zscore > 1).astype(int)
    cluster_intensity = high_volume.rolling(window=5, min_periods=3).sum()
    
    resistance_level = data['high'].rolling(window=10, min_periods=7).max()
    breakout_strength = (data['close'] - resistance_level) / resistance_level
    
    factor7 = cluster_intensity * breakout_strength
    
    # Amount-Flow Momentum Convergence
    current_flow = data['amount'] / data['volume']
    prev_flow = data['amount'].shift(1) / data['volume'].shift(1)
    flow_momentum = current_flow / prev_flow
    
    return_2d = data['close'] / data['close'].shift(2)
    return_5d = data['close'] / data['close'].shift(5)
    return_convergence = return_2d / return_5d
    
    factor8 = flow_momentum * return_convergence
    
    # Range-Volume Divergence Signal
    range_volume_ratio = ((data['high'] - data['low']) / data['close']) / data['volume']
    range_volume_mean_7d = range_volume_ratio.rolling(window=7, min_periods=5).mean()
    range_volume_divergence = range_volume_ratio / range_volume_mean_7d
    
    return_acceleration = (data['close'] / data['close'].shift(1) - 1) / (data['close'].shift(1) / data['close'].shift(2) - 1)
    momentum_stability = daily_return.rolling(window=4, min_periods=3).std()
    quality_score = return_acceleration / momentum_stability
    
    factor9 = range_volume_divergence * quality_score
    
    # Efficiency-Volume Regime Change
    efficiency_metric = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    efficiency_median_10d = efficiency_metric.rolling(window=10, min_periods=6).median()
    efficiency_shift = efficiency_metric / efficiency_median_10d
    
    volume_regime = data['volume'] / data['volume'].shift(1)
    volume_median = data['volume'].rolling(window=10, min_periods=6).median()
    volume_persistence = (data['volume'] > volume_median).rolling(window=5, min_periods=3).sum()
    
    factor10 = efficiency_shift * volume_regime * volume_persistence
    
    # Combine all factors with equal weights
    combined_factor = (
        factor1.fillna(0) + factor2.fillna(0) + factor3.fillna(0) + 
        factor4.fillna(0) + factor5.fillna(0) + factor6.fillna(0) + 
        factor7.fillna(0) + factor8.fillna(0) + factor9.fillna(0) + 
        factor10.fillna(0)
    )
    
    return combined_factor
