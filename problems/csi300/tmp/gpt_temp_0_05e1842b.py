import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Multi-timeframe Price Efficiency
    # 5-day price path efficiency ratio
    data['price_range_5d'] = data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()
    data['net_move_5d'] = data['close'] - data['close'].shift(5)
    data['total_move_5d'] = (data['close'] - data['open']).abs().rolling(window=5).sum()
    data['efficiency_5d'] = np.where(data['price_range_5d'] != 0, 
                                   data['net_move_5d'].abs() / data['price_range_5d'], 0)
    
    # 10-day price path efficiency ratio
    data['price_range_10d'] = data['high'].rolling(window=10).max() - data['low'].rolling(window=10).min()
    data['net_move_10d'] = data['close'] - data['close'].shift(10)
    data['total_move_10d'] = (data['close'] - data['open']).abs().rolling(window=10).sum()
    data['efficiency_10d'] = np.where(data['price_range_10d'] != 0, 
                                    data['net_move_10d'].abs() / data['price_range_10d'], 0)
    
    # Combined Fractal Momentum with Volume-weighted adjustment
    data['fractal_momentum'] = data['efficiency_5d'] * data['efficiency_10d']
    data['volume_weight'] = data['volume'] / data['volume'].rolling(window=20).mean()
    data['fractal_momentum_adj'] = data['fractal_momentum'] * data['volume_weight']
    
    # Regime Acceleration Analysis
    # Price Momentum Acceleration: 3-day minus 6-day price return difference
    data['return_3d'] = data['close'].pct_change(3)
    data['return_6d'] = data['close'].pct_change(6)
    data['price_acceleration'] = data['return_3d'] - data['return_6d']
    
    # Volume Regime Persistence
    # Volume clustering duration
    data['volume_ma_5'] = data['volume'].rolling(window=5).mean()
    data['volume_ma_20'] = data['volume'].rolling(window=20).mean()
    data['high_volume_regime'] = (data['volume_ma_5'] > data['volume_ma_20']).astype(int)
    
    # Calculate regime duration
    data['regime_duration'] = 0
    for i in range(1, len(data)):
        if data['high_volume_regime'].iloc[i] == data['high_volume_regime'].iloc[i-1]:
            data['regime_duration'].iloc[i] = data['regime_duration'].iloc[i-1] + 1
        else:
            data['regime_duration'].iloc[i] = 1
    
    # Volume acceleration pattern
    data['volume_change_3d'] = data['volume'].pct_change(3)
    data['volume_change_6d'] = data['volume'].pct_change(6)
    data['volume_acceleration'] = data['volume_change_3d'] - data['volume_change_6d']
    
    # Combine volume regime components
    data['volume_regime_score'] = (data['regime_duration'] * data['volume_acceleration']) / 10
    
    # Fractal-Regime Divergence Detection
    # Short-term vs Regime Signal Comparison
    data['short_term_signal'] = data['fractal_momentum_adj'] * np.sign(data['return_3d'])
    data['regime_signal'] = data['price_acceleration'] * data['volume_regime_score']
    
    # Multi-timeframe Consistency
    data['signal_alignment'] = np.sign(data['short_term_signal']) == np.sign(data['regime_signal'])
    data['alignment_strength'] = data['signal_alignment'].rolling(window=5).mean()
    
    # Final Alpha Composite
    # Combine fractal momentum with regime acceleration
    base_alpha = data['fractal_momentum_adj'] * data['price_acceleration'] * data['volume_regime_score']
    
    # Apply divergence-based weighting
    divergence_weight = 1 + (2 * data['alignment_strength'] - 1)
    final_alpha = base_alpha * divergence_weight
    
    return final_alpha
