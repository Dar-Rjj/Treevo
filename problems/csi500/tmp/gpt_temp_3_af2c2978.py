import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Momentum
    # Raw Intraday Return: Close(t) - Open(t)
    raw_return = data['close'] - data['open']
    
    # Normalized Momentum: Raw Return / (High(t) - Low(t))
    daily_range = data['high'] - data['low']
    daily_range = daily_range.replace(0, np.nan)  # Avoid division by zero
    normalized_momentum = raw_return / daily_range
    
    # Volume Confirmation
    # Volume Efficiency: Volume(t) / (High(t) - Low(t))
    volume_efficiency = data['volume'] / daily_range
    
    # Volume Ratio: Current Efficiency / Average(Efficiency(t-5:t-1))
    # Using rolling window with min_periods=1 to handle initial periods
    avg_efficiency = volume_efficiency.shift(1).rolling(window=5, min_periods=1).mean()
    volume_ratio = volume_efficiency / avg_efficiency
    volume_ratio = volume_ratio.replace([np.inf, -np.inf], np.nan)
    
    # Range Completion Pattern
    # Completion Ratio: (Close(t) - Low(t)) / (High(t) - Low(t))
    completion_ratio = (data['close'] - data['low']) / daily_range
    
    # Pattern Persistence: Correlation(Completion(t-4:t), Completion(t-9:t-5))
    # Calculate rolling correlations using only past data
    def calculate_persistence(series):
        if len(series) < 10:
            return np.nan
        recent = series.iloc[-5:]  # t-4:t
        previous = series.iloc[-10:-5]  # t-9:t-5
        return recent.corr(previous)
    
    pattern_persistence = completion_ratio.rolling(window=10, min_periods=10).apply(
        calculate_persistence, raw=False
    )
    
    # Factor Combination
    # Combined Alpha: Normalized Momentum × Volume Ratio × Pattern Persistence
    combined_alpha = normalized_momentum * volume_ratio * pattern_persistence
    
    return combined_alpha
