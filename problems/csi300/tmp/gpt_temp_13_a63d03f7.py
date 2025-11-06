import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum-Concentrated Pressure Component
    # Raw Intraday Pressure
    data['raw_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Multi-period Momentum
    data['momentum_5'] = data['close'] / data['close'].shift(5)
    data['momentum_10'] = data['close'] / data['close'].shift(10)
    data['momentum_combined'] = data['momentum_5'] * data['momentum_10']
    
    # Concentrated pressure
    data['concentrated_pressure'] = data['raw_pressure'] * data['momentum_combined']
    
    # Asymmetric Volume Confirmation
    # Calculate daily price change direction
    data['price_change'] = data['close'] - data['close'].shift(1)
    
    # Calculate upside and downside volumes for rolling window
    upside_volume_sum = 0
    downside_volume_sum = 0
    
    # Initialize arrays for rolling calculations
    upside_intensity = np.zeros(len(data))
    downside_intensity = np.zeros(len(data))
    
    for i in range(len(data)):
        if i >= 19:  # Need at least 20 days for calculations
            # Calculate upside volume (4-day window)
            upside_sum = 0
            downside_sum = 0
            for j in range(max(0, i-4), i+1):
                if data['price_change'].iloc[j] > 0:
                    upside_sum += data['volume'].iloc[j]
                elif data['price_change'].iloc[j] < 0:
                    downside_sum += data['volume'].iloc[j]
            
            # Calculate average volume (20-day window)
            avg_volume = data['volume'].iloc[max(0, i-19):i+1].mean()
            
            upside_intensity[i] = upside_sum / (avg_volume + 1e-8)
            downside_intensity[i] = downside_sum / (avg_volume + 1e-8)
    
    data['upside_intensity'] = upside_intensity
    data['downside_intensity'] = downside_intensity
    
    # Volume Asymmetry Ratio
    data['volume_asymmetry'] = data['upside_intensity'] / (data['downside_intensity'] + 1e-8)
    
    # Volume-Weighted Pressure Adjustment
    # Volume breakout confirmation
    data['volume_breakout'] = np.zeros(len(data))
    for i in range(len(data)):
        if i >= 19:
            avg_volume_20d = data['volume'].iloc[max(0, i-19):i+1].mean()
            data['volume_breakout'].iloc[i] = data['volume'].iloc[i] / (avg_volume_20d + 1e-8)
    
    # Weight concentrated pressure by volume asymmetry
    data['weighted_pressure'] = data['concentrated_pressure'] * data['volume_asymmetry'] * data['volume_breakout']
    
    # Efficiency Integration
    # Range Efficiency
    data['range_efficiency'] = (data['high'] - data['low']) / (data['close'].shift(1) + 1e-8)
    
    # Volume Efficiency
    data['volume_efficiency'] = data['volume'] / (data['high'] - data['low'] + 1e-8)
    
    # Combine all components with efficiency adjustments
    # Normalize by range efficiency and adjust by volume efficiency
    data['final_factor'] = (data['weighted_pressure'] / (data['range_efficiency'] + 1e-8)) * data['volume_efficiency']
    
    # Return the factor series
    return data['final_factor']
