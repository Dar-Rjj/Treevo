import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Multi-Timeframe Price-Volume-Liquidity Momentum Alignment Alpha Factor
    
    This factor captures the alignment of momentum across multiple timeframes,
    confirmed by volume signals and enhanced by liquidity quality indicators.
    """
    
    # Extract price and volume data
    close = data['close']
    high = data['high']
    low = data['low']
    open_price = data['open']
    volume = data['volume']
    amount = data['amount']
    
    # Initialize result series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # Multi-Timeframe Momentum Components
    momentum_3d = (close / close.shift(3)) - 1
    momentum_8d = (close / close.shift(8)) - 1
    momentum_13d = (close / close.shift(13)) - 1
    momentum_21d = (close / close.shift(21)) - 1
    
    # Volume Confirmation Signals
    # Volume Intensity Ratios
    volume_ma5 = volume.rolling(window=5).mean()
    volume_ma13 = volume.rolling(window=13).mean()
    volume_ma21 = volume.rolling(window=21).mean()
    
    volume_intensity_5 = volume / volume_ma5
    volume_intensity_13 = volume / volume_ma13
    volume_intensity_21 = volume / volume_ma21
    
    # Volume Trend Persistence
    volume_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(7, len(data)):
        window_start = max(0, i-7)
        window_end = i
        volume_window = volume.iloc[window_start:window_end+1]
        volume_ma21_window = volume_ma21.iloc[window_start:window_end+1]
        persistence_count = (volume_window > volume_ma21_window).sum()
        volume_persistence.iloc[i] = persistence_count
    
    # Liquidity Quality Indicators
    # Amount-Based Liquidity Ratios
    amount_ma5 = amount.rolling(window=5).mean()
    amount_ma13 = amount.rolling(window=13).mean()
    amount_ma21 = amount.rolling(window=21).mean()
    
    liquidity_5 = amount / amount_ma5
    liquidity_13 = amount / amount_ma13
    liquidity_21 = amount / amount_ma21
    
    # Price Efficiency Measures
    daily_range_position = (close - low) / (high - low).replace(0, np.nan)
    overnight_gap = (open_price - close.shift(1)) / close.shift(1)
    
    # Multiplicative Interaction Framework
    # Momentum Alignment Products
    momentum_alignment_ultra_long = momentum_3d * momentum_21d
    momentum_alignment_short_medium = momentum_8d * momentum_13d
    momentum_alignment_all = momentum_3d * momentum_8d * momentum_13d * momentum_21d
    
    # Volume-Momentum Alignment
    volume_momentum_short = momentum_8d * volume_intensity_13
    volume_momentum_medium = momentum_13d * volume_persistence
    
    # Liquidity-Momentum Integration
    liquidity_momentum_medium = momentum_13d * liquidity_21
    
    # Final Alpha Factor Construction
    # Primary Factor: Ultra-Short × Long-Term Momentum Alignment
    primary_factor = momentum_alignment_ultra_long
    
    # Enhanced Factor: Primary × Volume Confirmation
    enhanced_factor = primary_factor * volume_momentum_short
    
    # Complete Alpha Factor: Enhanced × Liquidity Quality
    complete_alpha = enhanced_factor * liquidity_momentum_medium
    
    # Handle NaN values by forward filling
    alpha = complete_alpha.fillna(method='ffill')
    
    return alpha
