import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Compute Intraday Price Velocity Components
    # Morning Momentum: (High[t] + Low[t])/2 - Open[t]
    morning_momentum = (data['high'] + data['low']) / 2 - data['open']
    
    # Afternoon Momentum: Close[t] - (High[t] + Low[t])/2
    afternoon_momentum = data['close'] - (data['high'] + data['low']) / 2
    
    # Momentum Divergence: Morning Momentum - Afternoon Momentum
    momentum_divergence = morning_momentum - afternoon_momentum
    
    # Intraday Range Acceleration
    current_range = data['high'] - data['low']
    lagged_range = current_range.shift(3)
    range_acceleration = (current_range - lagged_range) / lagged_range
    range_acceleration = range_acceleration.replace([np.inf, -np.inf], np.nan)
    
    # Compute Volume Flow Alignment Components
    # Volume Concentration Bias
    # Identify Up Days (Close > Previous Close) in past 10 days
    up_days = data['close'] > data['close'].shift(1)
    up_volume = data['volume'].rolling(window=10).apply(
        lambda x: np.sum(x[up_days.loc[x.index].values]) if up_days.loc[x.index].any() else 0, 
        raw=False
    )
    total_volume = data['volume'].rolling(window=10).sum()
    up_day_volume_ratio = up_volume / total_volume
    up_day_volume_ratio = up_day_volume_ratio.replace([np.inf, -np.inf], np.nan)
    
    # Volume Velocity Trend
    short_term_volume_velocity = data['volume'] / data['volume'].shift(5) - 1
    medium_term_volume_trend = data['volume'] / data['volume'].shift(10) - 1
    volume_trend_ratio = short_term_volume_velocity / medium_term_volume_trend
    volume_trend_ratio = volume_trend_ratio.replace([np.inf, -np.inf], np.nan)
    
    # Calculate Velocity-Volume Alignment Score
    # Combine Intraday Price Components
    combined_price_component = momentum_divergence * range_acceleration
    
    # Align with Volume Flow Components
    velocity_volume_alignment = (combined_price_component * 
                               up_day_volume_ratio * 
                               volume_trend_ratio)
    
    # Apply Intraday Reversal Enhancement
    # Morning vs Afternoon Return Product
    reversal_signal = morning_momentum * afternoon_momentum
    
    # Enhance Final Alignment Factor
    final_factor = velocity_volume_alignment * reversal_signal
    
    # Clean and return the factor
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan)
    return final_factor
