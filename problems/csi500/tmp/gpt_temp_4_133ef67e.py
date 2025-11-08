import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Asymmetry and Microstructure Factor
    Combines asymmetric volume impact, microstructure pressure, and temporal volume distribution
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize output series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # 1. Asymmetric Volume Impact
    # Up-Day Volume Concentration
    up_condition = data['close'] > data['open']
    up_volume_concentration = np.where(
        up_condition,
        ((data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)) * data['volume'],
        0
    )
    
    # Down-Day Volume Concentration
    down_condition = data['close'] < data['open']
    down_volume_concentration = np.where(
        down_condition,
        ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)) * data['volume'],
        0
    )
    
    # Volume Imbalance Persistence
    consecutive_up_volume = pd.Series(0, index=data.index)
    consecutive_down_volume = pd.Series(0, index=data.index)
    
    for i in range(1, len(data)):
        if data['close'].iloc[i] > data['close'].iloc[i-1] and data['volume'].iloc[i] > data['volume'].iloc[i-1]:
            consecutive_up_volume.iloc[i] = consecutive_up_volume.iloc[i-1] + 1
        if data['close'].iloc[i] < data['close'].iloc[i-1] and data['volume'].iloc[i] > data['volume'].iloc[i-1]:
            consecutive_down_volume.iloc[i] = consecutive_down_volume.iloc[i-1] + 1
    
    # Asymmetry Score
    asymmetry_score = (up_volume_concentration - down_volume_concentration) / (data['volume'] + 1e-8)
    
    # 2. Microstructure Pressure Indicators
    # Opening Gap Absorption
    opening_gap_absorption = pd.Series(0.0, index=data.index)
    for i in range(1, len(data)):
        gap = abs(data['open'].iloc[i] - data['close'].iloc[i-1])
        daily_range = data['high'].iloc[i] - data['low'].iloc[i]
        if daily_range > 0:
            opening_gap_absorption.iloc[i] = gap / daily_range
    
    # End-of-Day Pressure
    daily_mid = (data['high'] + data['low']) / 2
    daily_range = data['high'] - data['low'] + 1e-8
    end_of_day_pressure = abs(data['close'] - daily_mid) / daily_range
    
    # Microstructure Pressure Score
    microstructure_pressure_score = opening_gap_absorption + end_of_day_pressure
    
    # 3. Temporal Volume Distribution (simplified)
    # Using rolling windows to approximate temporal patterns
    morning_volume_ratio = data['volume'].rolling(window=5, min_periods=3).mean() / data['volume'].rolling(window=20, min_periods=10).mean()
    
    # Volume acceleration using rolling standard deviation
    volume_acceleration = data['volume'].rolling(window=10, min_periods=5).std() / (data['volume'].rolling(window=10, min_periods=5).mean() + 1e-8)
    
    # Volume Distribution Score
    volume_distribution_score = morning_volume_ratio * volume_acceleration
    
    # 4. Final Alpha Integration
    # Normalize components to avoid extreme values
    asymmetry_score_norm = (asymmetry_score - asymmetry_score.rolling(window=20, min_periods=10).mean()) / (asymmetry_score.rolling(window=20, min_periods=10).std() + 1e-8)
    microstructure_pressure_norm = (microstructure_pressure_score - microstructure_pressure_score.rolling(window=20, min_periods=10).mean()) / (microstructure_pressure_score.rolling(window=20, min_periods=10).std() + 1e-8)
    volume_distribution_norm = (volume_distribution_score - volume_distribution_score.rolling(window=20, min_periods=10).mean()) / (volume_distribution_score.rolling(window=20, min_periods=10).std() + 1e-8)
    
    # Final alpha factor
    alpha = asymmetry_score_norm * microstructure_pressure_norm * volume_distribution_norm
    
    # Handle NaN values
    alpha = alpha.fillna(0)
    
    return alpha
