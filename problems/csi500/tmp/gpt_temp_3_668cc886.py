import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Directional Volatility Analysis
    # Calculate upside volatility (close > open periods)
    upside_vol = np.where(data['close'] > data['open'], 
                         (data['high'] - data['low']) / data['open'], 
                         0)
    
    # Calculate downside volatility (close < open periods)
    downside_vol = np.where(data['close'] < data['open'], 
                           (data['high'] - data['low']) / data['open'], 
                           0)
    
    # Compute rolling directional volatilities (10-day window)
    upside_vol_rolling = pd.Series(upside_vol, index=data.index).rolling(window=10, min_periods=5).mean()
    downside_vol_rolling = pd.Series(downside_vol, index=data.index).rolling(window=10, min_periods=5).mean()
    
    # Compute volatility asymmetry ratio (upside/downside) with smoothing
    volatility_asymmetry = upside_vol_rolling / (downside_vol_rolling + 1e-8)
    volatility_asymmetry = volatility_asymmetry.rolling(window=5, min_periods=3).mean()
    
    # Breakout Detection
    # Identify 20-day high/low breaks
    high_20d = data['high'].rolling(window=20, min_periods=10).max()
    low_20d = data['low'].rolling(window=20, min_periods=10).min()
    
    # Calculate breakout strength (magnitude of break)
    high_breakout = np.where(data['close'] > high_20d.shift(1), 
                            (data['close'] - high_20d.shift(1)) / high_20d.shift(1), 
                            0)
    
    low_breakout = np.where(data['close'] < low_20d.shift(1), 
                           (low_20d.shift(1) - data['close']) / low_20d.shift(1), 
                           0)
    
    breakout_strength = high_breakout - low_breakout
    
    # Volume Confirmation
    # Compute 5-day volume trend
    volume_trend = data['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else 0
    )
    
    # Assess volume-breakout alignment
    volume_breakout_alignment = np.where(
        (breakout_strength > 0) & (volume_trend > 0), 1,
        np.where((breakout_strength < 0) & (volume_trend < 0), -1, 0)
    )
    
    # Signal Integration
    # Combine volatility asymmetry with breakout strength
    raw_signal = volatility_asymmetry * breakout_strength
    
    # Weight by volume confirmation
    volume_weight = np.where(volume_breakout_alignment != 0, 
                           np.abs(volume_trend) / (np.abs(volume_trend).rolling(window=20, min_periods=10).mean() + 1e-8), 
                           0.5)
    
    # Generate final breakout signal
    final_signal = raw_signal * volume_weight
    
    # Normalize the signal using rolling z-score (20-day window)
    signal_mean = final_signal.rolling(window=20, min_periods=10).mean()
    signal_std = final_signal.rolling(window=20, min_periods=10).std()
    normalized_signal = (final_signal - signal_mean) / (signal_std + 1e-8)
    
    return normalized_signal
