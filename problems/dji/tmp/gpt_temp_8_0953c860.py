import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Weighted High-Low Spread
    weighted_high_low_spread = (df['high'] - df['low']) * df['volume']
    
    # Apply Conditional Weight to High-Low Spread
    positive_return_weight = 1.5
    negative_return_weight = 0.5
    conditional_weight = np.where(df['close'] > df['open'], positive_return_weight, negative_return_weight)
    weighted_high_low_spread = weighted_high_low_spread * conditional_weight
    
    # Calculate Intraday Price Movement
    intraday_price_movement = df['close'] - df['open']
    
    # Determine Volume Increase from Average
    volume_increase = df['volume'] - df['volume'].rolling(window=5).mean()
    
    # Compute Intraday Volatility
    intraday_volatility = (df['high'] - df['low']) * df['volume']
    
    # Weight Intraday Price Movement by Volume Spike
    weighted_intraday_price_movement = intraday_price_movement * abs(volume_increase)
    
    # Calculate Daily Return
    daily_return = df['close'].pct_change()
    
    # Volume-Weighted Rolling Metrics
    volume_weighted_momentum = (daily_return * df['volume']).rolling(window=10).sum()
    volume_weighted_volatility = np.sqrt(((daily_return * np.sqrt(df['volume'])) ** 2).rolling(window=10).sum())
    
    # Momentum-Volatility Interaction
    momentum_volatility_interaction = volume_weighted_momentum * volume_weighted_volatility
    
    # Significant Daily Range
    significant_daily_range = df['high'] - df['low']
    
    # Weight by Volume Spike Magnitude
    weighted_ratio = (volume_increase * (intraday_price_movement / volume_increase)).fillna(0)
    
    # Incorporate Price Directionality
    price_directionality = np.where(intraday_price_movement > 0, 1, -1) * weighted_ratio
    
    # Integrate Price Volatility
    integrated_price_volatility = price_directionality / significant_daily_range.rolling(window=5).mean()
    
    # Cumulative Return
    cumulative_return = (df['close'] / df['close'].shift(10)) - 1
    
    # Adjust for Price Range
    volume_adjusted_spread = df['volume'] * (df['high'] - df['low'])
    normalized_high_low_spread = volume_adjusted_spread / (df['high'].rolling(window=10).max() - df['low'].rolling(window=10).min())
    
    # Final Alpha Factor Components
    alpha_factor = (
        cumulative_return * 
        momentum_volatility_interaction * 
        normalized_high_low_spread * 
        integrated_price_volatility
    )
    
    # Condition on Close-to-Open Return
    final_alpha_factor = np.where(df['close'] > df['open'], alpha_factor * positive_return_weight, alpha_factor * negative_return_weight)
    
    return final_alpha_factor
