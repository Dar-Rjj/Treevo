import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Divergence
    # Short-term (3-day) divergence
    short_price_return = data['close'] / data['close'].shift(3) - 1
    short_volume_return = data['volume'] / data['volume'].shift(3) - 1
    short_divergence = short_price_return - short_volume_return
    
    # Medium-term (10-day) divergence
    medium_price_return = data['close'] / data['close'].shift(10) - 1
    medium_volume_return = data['volume'] / data['volume'].shift(10) - 1
    medium_divergence = medium_price_return - medium_volume_return
    
    # Divergence Acceleration
    divergence_acceleration = short_divergence - medium_divergence
    
    # Volatility Context
    # Daily Range
    daily_range = (data['high'] - data['low']) / data['close']
    
    # Volatility Ratio
    vol_ratio = data['close'].rolling(5).std() / data['close'].rolling(10).std()
    
    # Range Efficiency
    range_efficiency = (data['close'] - data['open']) / (data['high'] - data['low'])
    range_efficiency = range_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # Pattern Detection
    # Failed Breakouts (count over 5 days)
    failed_breakout_condition = (data['high'] > data['high'].shift(1)) & (data['close'] < data['close'].shift(1))
    failed_breakouts = failed_breakout_condition.rolling(5, min_periods=1).sum()
    
    # Abnormal Volume
    volume_ma = data['volume'].rolling(10).mean().shift(1)
    abnormal_volume = data['volume'] / volume_volume_ma
    abnormal_volume = abnormal_volume.replace([np.inf, -np.inf], np.nan)
    
    # Regime-Adaptive Synthesis
    # Define volatility regime (using volatility ratio as proxy)
    high_vol_regime = vol_ratio > 1.0
    low_vol_regime = vol_ratio <= 1.0
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # High Volatility regime factor
    high_vol_factor = short_divergence * divergence_acceleration * abnormal_volume
    
    # Low Volatility regime factor
    low_vol_factor = medium_divergence * range_efficiency * (data['close'] - data['open'])
    
    # Combine regimes
    factor[high_vol_regime] = high_vol_factor[high_vol_regime]
    factor[low_vol_regime] = low_vol_factor[low_vol_regime]
    
    # Fill any remaining NaN values with 0
    factor = factor.fillna(0)
    
    return factor
