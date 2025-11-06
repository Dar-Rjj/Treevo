import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy the dataframe to avoid modifying the original
    data = df.copy()
    
    # Calculate returns
    data['returns'] = data['close'].pct_change()
    
    # Price Fractal Efficiency Calculation
    window = 10
    
    # Cumulative price movement (sum of absolute price changes)
    data['price_cumulative_movement'] = (data['high'] - data['low']).rolling(window=window).sum()
    
    # Straight-line distance between day t-10 and day t
    data['straight_line_distance'] = abs(data['close'] - data['close'].shift(window))
    
    # Price fractal efficiency ratio
    data['price_efficiency'] = data['straight_line_distance'] / data['price_cumulative_movement']
    data['price_efficiency'] = data['price_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Volume Fractal Patterns
    # Calculate signed volume (volume * sign of return)
    data['signed_volume'] = data['volume'] * np.sign(data['returns'])
    
    # Cumulative volume flow
    data['cumulative_volume'] = data['volume'].rolling(window=window).sum()
    
    # Net volume direction (sum of signed volume)
    data['net_volume_direction'] = data['signed_volume'].rolling(window=window).sum()
    
    # Volume efficiency ratio
    data['volume_efficiency'] = data['net_volume_direction'] / data['cumulative_volume']
    data['volume_efficiency'] = data['volume_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Regime Transition Detection
    # Efficiency momentum (change in fractal efficiency)
    data['price_efficiency_momentum'] = data['price_efficiency'].diff(3)
    data['volume_efficiency_momentum'] = data['volume_efficiency'].diff(3)
    
    # Detect efficiency regime shifts using rolling statistics
    data['price_efficiency_ma'] = data['price_efficiency'].rolling(window=5).mean()
    data['price_efficiency_std'] = data['price_efficiency'].rolling(window=5).std()
    
    # Identify regime transitions (when efficiency crosses its moving average significantly)
    data['price_regime_shift'] = np.where(
        abs(data['price_efficiency'] - data['price_efficiency_ma']) > data['price_efficiency_std'],
        1, 0
    )
    
    # Volume efficiency divergences
    data['volume_efficiency_ma'] = data['volume_efficiency'].rolling(window=5).mean()
    data['volume_regime_shift'] = np.where(
        abs(data['volume_efficiency'] - data['volume_efficiency_ma']) > data['price_efficiency_std'],
        1, 0
    )
    
    # Combined regime transition signal
    data['regime_transition'] = data['price_regime_shift'] | data['volume_regime_shift']
    
    # Construct Transition-Enhanced Alpha
    # Normalize efficiency metrics
    data['norm_price_eff'] = (data['price_efficiency'] - data['price_efficiency'].rolling(window=20).mean()) / data['price_efficiency'].rolling(window=20).std()
    data['norm_volume_eff'] = (data['volume_efficiency'] - data['volume_efficiency'].rolling(window=20).mean()) / data['volume_efficiency'].rolling(window=20).std()
    
    # Efficiency convergence/divergence signals
    data['efficiency_divergence'] = data['norm_price_eff'] - data['norm_volume_eff']
    
    # Apply regime transition weights
    transition_weight = 2.0  # Higher weight during transitions
    data['transition_multiplier'] = np.where(data['regime_transition'] == 1, transition_weight, 1.0)
    
    # Implement Dynamic Efficiency Scoring
    # Recent efficiency changes with exponential decay
    decay_factor = 0.8
    data['recent_price_eff_change'] = data['price_efficiency_momentum'].rolling(window=3).apply(
        lambda x: np.sum(x * np.array([decay_factor**i for i in range(len(x))][::-1]))
    )
    
    data['recent_volume_eff_change'] = data['volume_efficiency_momentum'].rolling(window=3).apply(
        lambda x: np.sum(x * np.array([decay_factor**i for i in range(len(x))][::-1]))
    )
    
    # Dynamic efficiency prediction factor
    data['transition_enhanced_alpha'] = (
        data['norm_price_eff'] * 0.4 + 
        data['norm_volume_eff'] * 0.3 +
        data['efficiency_divergence'] * 0.2 +
        data['recent_price_eff_change'] * 0.05 +
        data['recent_volume_eff_change'] * 0.05
    ) * data['transition_multiplier']
    
    # Final factor output
    factor = data['transition_enhanced_alpha']
    
    return factor
