import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate returns and daily range
    returns = df['close'].pct_change()
    daily_range = df['high'] - df['low']
    
    # Asymmetric Efficiency Component
    # Upside Efficiency Momentum
    def calc_upside_efficiency(window):
        positive_returns = returns.rolling(window).apply(
            lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else 0, raw=False
        )
        range_sum = daily_range.rolling(window).sum()
        return positive_returns / range_sum
    
    upside_short = calc_upside_efficiency(5)
    upside_medium = calc_upside_efficiency(10)
    upside_long = calc_upside_efficiency(20)
    
    # Downside Efficiency Momentum
    def calc_downside_efficiency(window):
        negative_returns = returns.rolling(window).apply(
            lambda x: x[x < 0].mean() if len(x[x < 0]) > 0 else 0, raw=False
        )
        range_sum = daily_range.rolling(window).sum()
        return negative_returns / range_sum
    
    downside_short = calc_downside_efficiency(5)
    downside_medium = calc_downside_efficiency(10)
    downside_long = calc_downside_efficiency(20)
    
    # Efficiency Asymmetry Ratio
    eff_asym_short = upside_short / downside_short.abs()
    eff_asym_medium = upside_medium / downside_medium.abs()
    eff_asym_long = upside_long / downside_long.abs()
    
    # Volume-Range Acceleration Component
    # Volume Acceleration Divergence
    volume_3d_change = df['volume'].pct_change(3)
    volume_8d_change = df['volume'].pct_change(8)
    volume_5d_change = df['volume'].pct_change(5)
    volume_20d_change = df['volume'].pct_change(20)
    
    vol_accel_short = volume_3d_change - volume_8d_change
    vol_accel_long = volume_5d_change - volume_20d_change
    vol_accel_divergence = vol_accel_short - vol_accel_long
    
    # Asymmetric Range Pressure
    avg_range_5d = daily_range.rolling(5).mean()
    upside_range_mom = (df['high'] - df['high'].shift(5)) / avg_range_5d
    downside_range_mom = (df['low'] - df['low'].shift(5)) / avg_range_5d
    range_asymmetry = upside_range_mom - downside_range_mom
    
    # Volume-Range Alignment
    vol_range_alignment = vol_accel_divergence * range_asymmetry
    
    # Multi-Scale Efficiency Divergence
    # Efficiency Acceleration
    short_medium_div = eff_asym_short / eff_asym_medium
    medium_long_div = eff_asym_medium / eff_asym_long
    combined_acceleration = short_medium_div * medium_long_div
    
    # Volatility Context
    range_volatility = daily_range.rolling(20).std()
    scaled_efficiency = combined_acceleration / range_volatility
    
    # Signal Integration
    factor_combination = scaled_efficiency * vol_range_alignment
    signal = np.arcsinh(factor_combination)
    
    return signal
