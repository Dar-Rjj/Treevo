import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Price Momentum
    data['high_to_close_momentum'] = data['high'] / data['close'] - 1
    data['low_to_close_momentum'] = data['low'] / data['close'] - 1
    data['intraday_range_efficiency'] = (data['high'] - data['low']) / data['close']
    
    # Volume Confirmation Patterns
    data['volume_5day_ma'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_trend_strength'] = data['volume'] / data['volume_5day_ma'] - 1
    
    # Volume Persistence (count consecutive days volume > 20-day MA)
    data['volume_20day_ma'] = data['volume'].rolling(window=20, min_periods=1).mean()
    data['volume_above_ma'] = data['volume'] > data['volume_20day_ma']
    
    # Calculate consecutive days count
    data['volume_persistence'] = 0
    current_streak = 0
    for i in range(len(data)):
        if data['volume_above_ma'].iloc[i]:
            current_streak += 1
        else:
            current_streak = 0
        data['volume_persistence'].iloc[i] = current_streak
    
    # Volume-Momentum Alignment
    momentum_diff = data['high_to_close_momentum'] - data['low_to_close_momentum']
    data['volume_momentum_alignment'] = np.sign(data['volume_trend_strength']) * np.sign(momentum_diff)
    
    # Price Movement Efficiency
    # True Range calculation
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    
    # Close Movement Efficiency
    data['close_movement_efficiency'] = abs(data['close'] - data['prev_close']) / data['true_range']
    
    # Intraday Efficiency
    data['intraday_efficiency'] = (data['high'] - data['low']) / data['true_range']
    
    # Composite Factor Generation
    # Volume-Confirmed Intraday Momentum
    data['volume_confirmed_intraday_momentum'] = (
        data['high_to_close_momentum'] - data['low_to_close_momentum']
    ) * data['volume_momentum_alignment']
    
    # Efficiency-Weighted Momentum
    data['efficiency_weighted_momentum'] = (
        data['volume_confirmed_intraday_momentum'] * data['close_movement_efficiency']
    )
    
    # Final Factor
    data['factor'] = data['efficiency_weighted_momentum'] * data['volume_trend_strength']
    
    # Return the factor series with proper indexing
    return data['factor']
