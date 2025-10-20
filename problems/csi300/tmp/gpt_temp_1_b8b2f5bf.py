import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Range Efficiency Analysis
    # Range calculations
    data['daily_range'] = data['high'] - data['low']
    data['daily_return'] = data['close'] - data['open']
    
    # Multi-timeframe range efficiency
    for window in [5, 10, 20]:
        data[f'range_eff_{window}d'] = (data['close'] / data['close'].shift(window) - 1) / (
            data['daily_range'].rolling(window).mean() / data['close'].shift(window))
    
    # Range efficiency acceleration
    data['range_eff_accel_primary'] = data['range_eff_5d'] - data['range_eff_10d']
    data['range_eff_accel_secondary'] = data['range_eff_10d'] - data['range_eff_20d']
    
    # Range utilization efficiency
    data['range_utilization'] = data['daily_return'] / data['daily_range']
    data['range_util_momentum'] = data['range_utilization'] - data['range_utilization'].shift(5)
    
    # Volume-Amount Flow Efficiency Momentum
    data['volume_efficiency'] = data['volume'] / data['daily_range']
    data['amount_efficiency'] = data['amount'] / data['daily_range']
    
    # Volume efficiency momentum and acceleration
    data['volume_eff_momentum'] = data['volume_efficiency'] - data['volume_efficiency'].shift(5)
    data['amount_eff_momentum'] = data['amount_efficiency'] - data['amount_efficiency'].shift(5)
    
    # Volume-amount flow convergence
    data['volume_amount_convergence'] = data['volume_eff_momentum'] * data['amount_eff_momentum']
    
    # Position-Gap Efficiency Context
    data['position_efficiency'] = (data['close'] - data['low']) / data['daily_range']
    data['gap_efficiency'] = (data['open'] - data['close'].shift(1)) / data['daily_range']
    
    # Position efficiency momentum
    data['position_eff_momentum'] = data['position_efficiency'] - data['position_efficiency'].shift(5)
    data['gap_eff_momentum'] = data['gap_efficiency'].rolling(3).sum()
    
    # Multi-Dimensional Efficiency Divergence Detection
    # Range-volume-amount divergence
    data['range_volume_div'] = data['range_eff_5d'] - data['volume_eff_momentum']
    data['range_amount_div'] = data['range_eff_5d'] - data['amount_eff_momentum']
    data['volume_amount_div'] = data['volume_eff_momentum'] - data['amount_eff_momentum']
    
    # Position-gap divergence
    data['position_gap_div'] = data['position_eff_momentum'] - data['gap_eff_momentum']
    data['position_range_div'] = data['position_eff_momentum'] - data['range_eff_5d']
    
    # Multi-timeframe divergence
    data['multi_timeframe_div'] = data['range_eff_accel_primary'] - data['range_eff_accel_secondary']
    
    # Efficiency Persistence & Momentum Strength
    # Range efficiency persistence (simplified as rolling correlation)
    data['range_eff_persistence'] = data['range_eff_5d'].rolling(5).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) == 5 and not np.isnan(x).any() else 0)
    
    # Volume-amount persistence alignment
    data['volume_amount_persistence'] = data['volume_efficiency'].rolling(5).std() / data['amount_efficiency'].rolling(5).std()
    
    # Adaptive Multi-Dimensional Signal Generation
    # Core efficiency divergence construction
    data['base_divergence'] = (
        data['range_volume_div'] + 
        data['range_amount_div'] + 
        data['volume_amount_div'] +
        data['position_gap_div']
    ) / 4
    
    # Volume-amount flow confirmation multiplier
    data['flow_confirmation'] = np.tanh(data['volume_amount_convergence'] * 0.01)
    
    # Position-gap efficiency context adjustment
    data['position_gap_context'] = np.tanh(data['position_gap_div'] * 0.1)
    
    # Range-weighted efficiency scaling
    data['range_weight'] = data['daily_range'] / data['daily_range'].rolling(20).mean()
    
    # Efficiency momentum enhancement
    data['momentum_enhancement'] = (
        data['range_eff_accel_primary'] + 
        data['volume_eff_momentum'] + 
        data['amount_eff_momentum']
    ) / 3
    
    # Final adaptive alpha factor
    # Multiplicative combination with range weighting
    alpha_factor = (
        data['base_divergence'] * 
        (1 + data['flow_confirmation']) * 
        (1 + data['position_gap_context']) * 
        data['range_weight'] * 
        (1 + np.tanh(data['momentum_enhancement'] * 0.1))
    )
    
    # Clean up and return
    alpha_series = alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    return alpha_series
