import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Momentum Component
    # 5-day Price Momentum
    data['price_momentum'] = data['close'] / data['close'].shift(5) - 1
    
    # Volatility-Adjusted Momentum
    data['volatility_adjusted_momentum'] = data['price_momentum'] / (data['high'] - data['low'])
    
    # Momentum Persistence - count consecutive days with same momentum direction
    data['momentum_direction'] = np.sign(data['price_momentum'])
    data['momentum_persistence'] = 0
    current_direction = None
    current_streak = 0
    
    for i in range(len(data)):
        if pd.isna(data['momentum_direction'].iloc[i]):
            data.loc[data.index[i], 'momentum_persistence'] = 0
            continue
            
        if current_direction is None:
            current_direction = data['momentum_direction'].iloc[i]
            current_streak = 1
        elif data['momentum_direction'].iloc[i] == current_direction:
            current_streak += 1
        else:
            current_direction = data['momentum_direction'].iloc[i]
            current_streak = 1
        
        data.loc[data.index[i], 'momentum_persistence'] = current_streak
    
    # Volume Efficiency Analysis
    # Volume per Price Movement
    data['volume_per_price_movement'] = data['volume'] / np.abs(data['close'] - data['close'].shift(1))
    data['volume_per_price_movement'] = data['volume_per_price_movement'].replace([np.inf, -np.inf], np.nan)
    
    # Volume-to-Range Ratio
    data['volume_to_range_ratio'] = data['volume'] / (data['high'] - data['low'])
    data['volume_to_range_ratio'] = data['volume_to_range_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Volume Efficiency Trend - 3-day slope using linear regression
    data['volume_efficiency_trend'] = np.nan
    for i in range(2, len(data)):
        if i >= 2:
            window_data = data['volume_per_price_movement'].iloc[i-2:i+1]
            if not window_data.isna().any():
                x = np.array([0, 1, 2])
                y = window_data.values
                slope = ((len(x) * np.sum(x*y) - np.sum(x) * np.sum(y)) / 
                        (len(x) * np.sum(x*x) - np.sum(x)**2))
                data.loc[data.index[i], 'volume_efficiency_trend'] = slope
    
    # Divergence Detection
    # Price-Volume Direction Divergence
    data['price_volume_divergence'] = np.sign(data['price_momentum']) * np.sign(data['volume_efficiency_trend'])
    
    # Divergence Strength
    data['divergence_strength'] = np.abs(data['price_momentum']) - np.abs(data['volume_efficiency_trend'])
    
    # Signal Generation
    # Base factor combining momentum, persistence, and divergence
    data['factor'] = (
        data['volatility_adjusted_momentum'] * 
        data['momentum_persistence'] * 
        (1 + data['divergence_strength']) * 
        np.sign(data['price_volume_divergence'])
    )
    
    # Enhanced signal with volume efficiency confirmation
    volume_confirmation = np.where(
        (data['volume_efficiency_trend'] > 0) & (data['price_momentum'] > 0) |
        (data['volume_efficiency_trend'] < 0) & (data['price_momentum'] < 0),
        1.2,  # Volume confirms price movement
        0.8   # Volume contradicts price movement
    )
    
    data['final_factor'] = data['factor'] * volume_confirmation
    
    # Handle NaN values
    data['final_factor'] = data['final_factor'].fillna(0)
    
    return data['final_factor']
