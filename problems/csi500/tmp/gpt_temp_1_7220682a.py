import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum-Convergence with Volume-Weighted Persistence factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Multi-Timeframe Momentum
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_8d'] = data['close'] / data['close'].shift(8) - 1
    data['momentum_13d'] = data['close'] / data['close'].shift(13) - 1
    
    # Detect Momentum Convergence
    data['momentum_diff_8_3'] = data['momentum_8d'] - data['momentum_3d']
    data['momentum_diff_13_8'] = data['momentum_13d'] - data['momentum_8d']
    
    # Calculate convergence strength (decreasing differences indicate convergence)
    data['convergence_strength'] = (
        -data['momentum_diff_8_3'].rolling(window=5, min_periods=3).mean() + 
        -data['momentum_diff_13_8'].rolling(window=5, min_periods=3).mean()
    )
    
    # Calculate Intraday Strength Persistence
    data['daily_range'] = data['high'] - data['low']
    data['close_position'] = (data['close'] - data['open']) / np.where(data['daily_range'] == 0, 1, data['daily_range'])
    
    # Calculate persistence pattern
    data['strong_day'] = (data['close'] > (data['high'] + data['low']) / 2).astype(int)
    
    # Count consecutive strong days
    data['persistence_count'] = 0
    current_streak = 0
    for i in range(len(data)):
        if data['strong_day'].iloc[i] == 1:
            current_streak += 1
        else:
            current_streak = 0
        data.iloc[i, data.columns.get_loc('persistence_count')] = current_streak
    
    # Calculate persistence magnitude
    data['avg_strength'] = data['close_position'].rolling(window=5, min_periods=3).mean()
    data['persistence_magnitude'] = data['persistence_count'] * data['avg_strength']
    
    # Combine with Volume Validation
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_ma_10'] = data['volume'].rolling(window=10, min_periods=5).mean()
    data['volume_acceleration'] = data['volume_ma_5'] / data['volume_ma_10'] - 1
    
    # Volatility awareness
    data['recent_volatility'] = (data['high'] - data['low']).rolling(window=10, min_periods=5).std()
    
    # Generate Final Factor
    # Weight convergence by volume acceleration and persistence
    base_factor = data['convergence_strength'] * data['persistence_magnitude']
    
    # Apply volume validation and volatility adjustment
    volume_weight = 1 + data['volume_acceleration']
    volatility_adjustment = 1 / (1 + data['recent_volatility'] / data['close'].rolling(window=20, min_periods=10).std())
    
    final_factor = base_factor * volume_weight * volatility_adjustment
    
    # Clean and return
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan)
    return final_factor
