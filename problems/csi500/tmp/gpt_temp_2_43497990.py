import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Divergence Analysis
    data['short_momentum'] = data['close'].pct_change(periods=3)
    data['medium_momentum'] = data['close'].pct_change(periods=10)
    data['momentum_divergence'] = data['short_momentum'] - data['medium_momentum']
    
    # Calculate divergence persistence
    data['divergence_direction'] = np.sign(data['momentum_divergence'])
    data['divergence_persistence'] = 0
    for i in range(1, len(data)):
        if data['divergence_direction'].iloc[i] == data['divergence_direction'].iloc[i-1]:
            data['divergence_persistence'].iloc[i] = data['divergence_persistence'].iloc[i-1] + 1
        else:
            data['divergence_persistence'].iloc[i] = 0
    
    # Volume-Elasticity Confirmation
    data['price_range'] = data['high'] - data['low']
    data['sma_price_range'] = data['price_range'].rolling(window=3).mean()
    data['price_elasticity'] = data['price_range'] / data['sma_price_range'] - 1
    
    data['volume_momentum'] = data['volume'] / data['volume'].shift(3)
    
    # Calculate volume-elasticity correlation (3-day rolling)
    data['volume_elasticity_corr'] = data['volume'].rolling(window=3).corr(data['price_elasticity'])
    
    # Identify confirmation signals
    data['confirmation_signal'] = ((data['momentum_divergence'] > 0) & (data['price_elasticity'] > 0)) | \
                                  ((data['momentum_divergence'] < 0) & (data['price_elasticity'] < 0))
    
    # Market Depth Integration
    data['amount_per_trade'] = data['amount'] / data['volume']
    
    # Calculate large trade intensity (assuming trades > 100k in amount are large)
    data['large_trade_count'] = (data['amount'] > 100000).rolling(window=5, min_periods=1).sum()
    data['total_trade_count'] = data['volume'].rolling(window=5, min_periods=1).count()
    data['large_trade_intensity'] = data['large_trade_count'] / data['total_trade_count']
    
    # Track institutional persistence
    data['high_intensity'] = data['large_trade_intensity'] > data['large_trade_intensity'].rolling(window=10).quantile(0.7)
    data['institutional_persistence'] = 0
    for i in range(1, len(data)):
        if data['high_intensity'].iloc[i]:
            data['institutional_persistence'].iloc[i] = data['institutional_persistence'].iloc[i-1] + 1
        else:
            data['institutional_persistence'].iloc[i] = 0
    
    # Depth validation
    data['depth_validation'] = data['large_trade_intensity'] * data['price_elasticity']
    
    # Adaptive Signal Generation
    data['base_signal'] = data['momentum_divergence'] * data['volume_momentum'] * data['price_elasticity']
    
    # Apply depth filter
    depth_threshold = data['large_trade_intensity'].rolling(window=20).quantile(0.6)
    data['depth_filtered_signal'] = data['base_signal'] * (data['large_trade_intensity'] > depth_threshold)
    
    # Extreme adjustment
    extreme_condition = abs(data['price_elasticity']) > 1.5
    data['extreme_adjusted_signal'] = np.where(
        extreme_condition,
        data['depth_filtered_signal'] * (1 - abs(data['price_elasticity'])),
        data['depth_filtered_signal']
    )
    
    # Calculate final signal strength
    persistence_weight = np.minimum(data['divergence_persistence'] / 5, 1.0)
    institutional_weight = np.minimum(data['institutional_persistence'] / 3, 1.0)
    
    data['final_signal'] = data['extreme_adjusted_signal'] * (1 + 0.2 * persistence_weight + 0.1 * institutional_weight)
    
    # Handle NaN values
    data['final_signal'] = data['final_signal'].fillna(0)
    
    return data['final_signal']
