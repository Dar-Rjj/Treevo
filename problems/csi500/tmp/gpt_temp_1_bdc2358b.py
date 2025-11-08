import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Acceleration with Volume-Volatility Confirmation
    # Multi-timeframe Price Momentum
    data['price_return_5d'] = data['close'] / data['close'].shift(5) - 1
    data['price_return_20d'] = data['close'] / data['close'].shift(20) - 1
    data['momentum_acceleration_ratio'] = data['price_return_5d'] / data['price_return_20d']
    
    # Volume Momentum Confirmation
    data['volume_return_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_return_20d'] = data['volume'] / data['volume'].shift(20) - 1
    data['volume_momentum_ratio'] = data['volume_return_5d'] / data['volume_return_20d']
    
    # Combine Price and Volume Acceleration
    data['combined_acceleration'] = data['momentum_acceleration_ratio'] * data['volume_momentum_ratio']
    
    # Volatility Context Adjustment
    data['daily_range'] = data['high'] - data['low']
    data['smoothed_volatility'] = data['daily_range'].rolling(window=10).mean()
    
    # Enhanced Momentum Factor
    data['momentum_factor'] = data['combined_acceleration'] * data['smoothed_volatility']
    
    # Intraday Strength with Liquidity-Volume Convergence
    # Intraday Momentum Strength
    data['daily_range_strength'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Persistence Pattern
    data['midpoint'] = (data['high'] + data['low']) / 2
    data['strong_day'] = (data['close'] > data['midpoint']).astype(int)
    
    # Calculate consecutive strong days
    data['consecutive_strong'] = 0
    current_streak = 0
    for i in range(len(data)):
        if data['strong_day'].iloc[i] == 1:
            current_streak += 1
        else:
            current_streak = 0
        data['consecutive_strong'].iloc[i] = current_streak
    
    data['avg_strength_5d'] = data['daily_range_strength'].rolling(window=5).mean()
    data['strength_persistence'] = data['consecutive_strong'] * data['avg_strength_5d']
    
    # Volume-Weighted Validation
    data['volume_efficiency'] = data['volume'] / (data['high'] - data['low'])
    
    # VWAP Convergence Pattern
    data['vwap'] = (data['high'] + data['low'] + data['close']) * data['volume'] / (3 * data['volume'])
    data['avg_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['vwap_convergence'] = data['vwap'] / data['avg_price']
    
    # Adjust Strength by Volume Context
    data['intraday_factor'] = data['strength_persistence'] * data['volume_efficiency'] * data['vwap_convergence']
    
    # Price-Range Convergence with Momentum Divergence
    # Price-Level Convergence
    data['high_10d'] = data['high'].rolling(window=10).max()
    data['low_10d'] = data['low'].rolling(window=10).min()
    data['range_compression'] = (data['high_10d'] - data['low_10d']) / data['close']
    data['range_position'] = (data['close'] - data['low_10d']) / (data['high_10d'] - data['low_10d'])
    
    # Momentum Divergence Validation
    data['price_momentum_divergence'] = data['price_return_5d'] / data['price_return_20d']
    data['volume_momentum_divergence'] = data['volume_return_5d'] / data['volume_return_20d']
    
    # Combine Range and Momentum Signals
    data['convergence_factor'] = data['range_position'] * data['price_momentum_divergence'] * data['volume_momentum_divergence']
    
    # Final Composite Factor
    # Combine all three components with equal weighting
    data['final_factor'] = (
        data['momentum_factor'].fillna(0) + 
        data['intraday_factor'].fillna(0) + 
        data['convergence_factor'].fillna(0)
    ) / 3
    
    return data['final_factor']
