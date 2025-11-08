import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Rejection Component
    intraday_rejection = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Volatility Asymmetry Weighting
    current_range = df['high'] - df['low']
    prev_range = df['high'].shift(1) - df['low'].shift(1)
    range_ratio = (current_range / prev_range.replace(0, np.nan)) - 1
    
    # Gap Direction Sign
    gap_direction = np.sign(df['close'].shift(1) - df['open'])
    
    # Volume Spike Asymmetry
    volume_median_5d = df['volume'].rolling(window=5, min_periods=1).median()
    volume_spike = df['volume'] / volume_median_5d.replace(0, np.nan)
    
    # Momentum Divergence
    short_momentum = df['close'] - df['close'].shift(5)
    medium_momentum = df['close'] - df['close'].shift(20)
    momentum_divergence = np.abs(short_momentum - medium_momentum)
    
    # Directional Volume Pressure
    directional_volume = df['volume'] * np.sign(df['close'] - df['open'])
    
    # Historical Pattern Component - 3-day Rolling Reversal
    returns_3d = df['close'].pct_change().rolling(window=3, min_periods=1)
    reversal_pattern = returns_3d.apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 and not np.all(x == x[0]) else 0, raw=True)
    
    # Volume Acceleration
    volume_acceleration = df['volume'] / df['volume'].shift(1).replace(0, np.nan)
    
    # Combine components
    price_rejection_component = intraday_rejection * range_ratio * gap_direction
    volume_momentum_component = volume_spike * momentum_divergence * directional_volume
    historical_pattern_component = reversal_pattern * volume_acceleration
    
    # Final factor calculation
    factor = price_rejection_component * volume_momentum_component * historical_pattern_component
    
    return factor
