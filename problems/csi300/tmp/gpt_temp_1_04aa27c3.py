import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Horizon Momentum-Volume Convergence Alpha Factor
    
    This factor combines:
    - Multi-timeframe momentum convergence patterns
    - Volatility-adjusted volume dynamics
    - Price-volume divergence with soft thresholds
    """
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Framework
    data['momentum_1d'] = data['close'] / data['close'].shift(1) - 1
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    
    # Momentum convergence scoring
    def calculate_convergence_score(row):
        positive_count = sum([1 for mom in [row['momentum_1d'], row['momentum_3d'], row['momentum_5d']] 
                            if mom > 0])
        return positive_count
    
    data['momentum_convergence_score'] = data.apply(calculate_convergence_score, axis=1)
    
    # Volatility-Adjusted Volume Dynamics
    data['volume_accel_1d'] = data['volume'] / data['volume'].shift(1)
    data['volume_accel_3d'] = data['volume'] / data['volume'].shift(3)
    data['volatility'] = (data['high'] - data['low']) / data['close']
    
    # Volume-volatility interaction multiplier
    def calculate_vol_vol_multiplier(row):
        vol_median = data['volume'].median()
        vol_threshold = vol_median
        vol_high = row['volume'] > vol_threshold
        
        vol_median = data['volatility'].median()
        vol_threshold = vol_median
        vol_high_vol = row['volatility'] > vol_threshold
        
        if vol_high and not vol_high_vol:
            return 1.3
        elif vol_high and vol_high_vol:
            return 1.1
        elif not vol_high and not vol_high_vol:
            return 0.9
        else:  # not vol_high and vol_high_vol
            return 0.7
    
    data['vol_vol_multiplier'] = data.apply(calculate_vol_vol_multiplier, axis=1)
    
    # Price-Volume Divergence with Soft Thresholds
    data['price_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['volume_momentum'] = data['volume'] / data['volume'].shift(1) - 1
    data['raw_divergence'] = data['price_momentum'] - data['volume_momentum']
    
    # Soft threshold application
    def calculate_divergence_weight(row):
        divergence = row['raw_divergence']
        if divergence > 0.03:
            return 1.4
        elif divergence > 0.01:
            return 1.2
        elif divergence >= -0.01:
            return 1.0
        elif divergence >= -0.03:
            return 0.8
        else:
            return 0.6
    
    data['divergence_weight'] = data.apply(calculate_divergence_weight, axis=1)
    
    # Factor Integration Logic
    # Base momentum signal (weighted average)
    data['base_momentum'] = (data['momentum_1d'] * 0.4 + 
                           data['momentum_3d'] * 0.35 + 
                           data['momentum_5d'] * 0.25)
    
    # Apply volume-volatility scaling
    data['scaled_momentum'] = data['base_momentum'] * data['vol_vol_multiplier']
    
    # Apply divergence adjustment
    data['divergence_adjusted'] = data['scaled_momentum'] * data['divergence_weight']
    
    # Apply convergence pattern emphasis
    data['final_factor'] = data['divergence_adjusted'] * (data['momentum_convergence_score'] + 1)
    
    # Return the final factor series
    return data['final_factor']
