import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Timeframe Efficiency Calculation
    df = df.copy()
    
    # Short-Term Efficiency (5-day)
    close_diff_5 = df['close'] - df['close'].shift(5)
    abs_close_diff_1 = abs(df['close'] - df['close'].shift(1))
    sum_abs_diff_5 = abs_close_diff_1.rolling(window=5, min_periods=5).sum()
    short_term_efficiency = close_diff_5 / sum_abs_diff_5
    
    # Medium-Term Efficiency (15-day)
    close_diff_15 = df['close'] - df['close'].shift(15)
    sum_abs_diff_15 = abs_close_diff_1.rolling(window=15, min_periods=15).sum()
    medium_term_efficiency = close_diff_15 / sum_abs_diff_15
    
    # Efficiency Divergence
    efficiency_divergence = short_term_efficiency - medium_term_efficiency
    
    # Volatility Context
    range_volatility = (df['high'] - df['low']).rolling(window=20, min_periods=20).std()
    return_volatility = (df['close'] / df['close'].shift(1) - 1).rolling(window=20, min_periods=20).std()
    combined_volatility = np.sqrt(range_volatility * return_volatility)
    
    # Base Efficiency Signal
    base_efficiency_signal = efficiency_divergence / combined_volatility
    
    # Breakout Identification
    high_20_max = df['high'].rolling(window=20, min_periods=20).max()
    low_20_min = df['low'].rolling(window=20, min_periods=20).min()
    
    high_breakout = (df['high'] > high_20_max) & (df['high'] > df['high'].shift(1))
    low_breakout = (df['low'] < low_20_min) & (df['low'] < df['low'].shift(1))
    
    # Breakout Strength
    high_breakout_strength = np.where(high_breakout, 
                                    (df['high'] - high_20_max) / (df['high'] - df['low']), 0)
    low_breakout_strength = np.where(low_breakout, 
                                   (low_20_min - df['low']) / (df['high'] - df['low']), 0)
    breakout_strength = high_breakout_strength + low_breakout_strength
    
    # Momentum Acceleration
    momentum_5_day = df['close'] / df['close'].shift(5) - 1
    momentum_10_day = df['close'].shift(5) / df['close'].shift(10) - 1
    momentum_acceleration = momentum_5_day - momentum_10_day
    
    # Acceleration Persistence
    acceleration_sign = np.sign(momentum_acceleration)
    acceleration_persistence = acceleration_sign.copy()
    for i in range(3, len(df)):
        if i >= 3:
            current_sign = acceleration_sign.iloc[i]
            same_sign_count = sum(acceleration_sign.iloc[i-2:i+1] == current_sign)
            acceleration_persistence.iloc[i] = current_sign * same_sign_count
    
    # Acceleration-Weighted Breakout
    acceleration_weighted_breakout = breakout_strength * momentum_acceleration * acceleration_persistence
    
    # VWAP Deviation Analysis
    vwap = (df['high'] + df['low'] + df['close']) / 3
    vwap_deviation = (df['close'] - vwap) / abs(df['close'])
    vwap_trend = (vwap_deviation - vwap_deviation.shift(5)) / abs(vwap_deviation.shift(5))
    
    # Volume Dynamics
    volume_ma_5 = df['volume'].rolling(window=5, min_periods=5).mean()
    volume_ma_15 = df['volume'].rolling(window=15, min_periods=15).mean()
    volume_trend = volume_ma_5 / volume_ma_15 - 1
    
    volume_ma_20 = df['volume'].rolling(window=20, min_periods=20).mean()
    volume_breakout_alignment = ((high_breakout | low_breakout) & (df['volume'] > volume_ma_20)).astype(float)
    
    # Volume-Efficiency Correlation
    volume_efficiency_corr = pd.Series(index=df.index, dtype=float)
    for i in range(10, len(df)):
        if i >= 10:
            window_data = df.iloc[i-9:i+1]
            corr = window_data['volume'].corr(abs(window_data['close'] - window_data['close'].shift(1)))
            volume_efficiency_corr.iloc[i] = corr if not np.isnan(corr) else 0
    
    # Volume Confirmation Score
    volume_confirmation_score = volume_trend * volume_breakout_alignment * volume_efficiency_corr
    
    # Final Alpha
    final_alpha = base_efficiency_signal * acceleration_weighted_breakout * volume_confirmation_score * np.sign(vwap_trend)
    
    return final_alpha
