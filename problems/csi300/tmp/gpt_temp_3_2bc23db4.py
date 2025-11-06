import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original DataFrame
    data = df.copy()
    
    # Fractal Price Structure Analysis
    # Multi-Scale Price Momentum
    data['short_momentum'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    data['medium_momentum'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    data['long_momentum'] = (data['close'] - data['close'].shift(6)) / data['close'].shift(6)
    
    # Price Range Dynamics
    data['intraday_range_eff'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['overnight_range_eff'] = (data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    
    # Volume Fractal Architecture
    # Volume Momentum Structure
    data['volume_acceleration'] = (data['volume'] / data['volume'].shift(1)) - (data['volume'].shift(1) / data['volume'].shift(2))
    data['volume_persistence'] = (data['volume'] / data['volume'].shift(3)) - (data['volume'].shift(3) / data['volume'].shift(6))
    
    # Volume Distribution Analysis
    data['volume_concentration'] = data['volume'] / (data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3))
    data['volume_dispersion'] = abs(data['volume'] - data['volume'].shift(1)) / (data['volume'] + data['volume'].shift(1))
    
    # Asymmetric Regime Detection
    # Price Regime Classification
    abs_short = abs(data['short_momentum'])
    abs_medium = abs(data['medium_momentum'])
    abs_long = abs(data['long_momentum'])
    
    data['trend_regime'] = (abs_short > abs_medium) & (abs_medium > abs_long)
    data['mean_reversion_regime'] = (abs_short < abs_medium) & (abs_medium < abs_long)
    
    # Volume Regime Classification
    data['high_volume_regime'] = (data['volume'] > data['volume'].shift(1)) & (data['volume'] > data['volume'].shift(2)) & (data['volume'] > data['volume'].shift(3))
    data['low_volume_regime'] = (data['volume'] < data['volume'].shift(1)) & (data['volume'] < data['volume'].shift(2)) & (data['volume'] < data['volume'].shift(3))
    
    # Fractal Momentum Integration
    # Multi-Scale Momentum Convergence
    data['short_medium_conv'] = data['short_momentum'] * data['medium_momentum']
    data['medium_long_conv'] = data['medium_momentum'] * data['long_momentum']
    
    # Volume-Enhanced Momentum
    data['volume_weighted_momentum'] = data['short_momentum'] * data['volume_concentration']
    data['acceleration_enhanced_momentum'] = (data['short_momentum'] - data['medium_momentum']) * data['volume_acceleration']
    
    # Asymmetric Factor Construction
    # Trend Regime Factors
    data['trend_high_volume'] = data['short_medium_conv'] * data['volume_weighted_momentum'] * data['intraday_range_eff']
    data['trend_low_volume'] = data['acceleration_enhanced_momentum'] * data['volume_persistence'] * data['overnight_range_eff']
    
    # Mean Reversion Regime Factors
    data['mr_high_volume'] = data['medium_long_conv'] * data['volume_dispersion'] * data['volume_acceleration']
    data['mr_low_volume'] = (data['short_momentum'] - data['medium_momentum']) * data['volume_concentration'] * data['intraday_range_eff']
    
    # Regime-Adaptive Factor Selection
    factor = pd.Series(index=data.index, dtype=float)
    
    # Primary Regime Detection and Volume Regime Sub-selection
    for idx in data.index:
        if data.loc[idx, 'trend_regime']:
            if data.loc[idx, 'high_volume_regime']:
                selected_factor = data.loc[idx, 'trend_high_volume']
            elif data.loc[idx, 'low_volume_regime']:
                selected_factor = data.loc[idx, 'trend_low_volume']
            else:
                # Default to trend high volume if no clear volume regime
                selected_factor = data.loc[idx, 'trend_high_volume']
        elif data.loc[idx, 'mean_reversion_regime']:
            if data.loc[idx, 'high_volume_regime']:
                selected_factor = data.loc[idx, 'mr_high_volume']
            elif data.loc[idx, 'low_volume_regime']:
                selected_factor = data.loc[idx, 'mr_low_volume']
            else:
                # Default to mean reversion high volume if no clear volume regime
                selected_factor = data.loc[idx, 'mr_high_volume']
        else:
            # Default case - use weighted combination
            selected_factor = (data.loc[idx, 'trend_high_volume'] + data.loc[idx, 'mr_high_volume']) / 2
        
        # Final Factor Output with volume acceleration adjustment
        volume_acc_adj = 1 + (data.loc[idx, 'volume_acceleration'] * 0.1)
        factor.loc[idx] = selected_factor * volume_acc_adj
    
    # Handle NaN values that may occur due to shifts
    factor = factor.fillna(0)
    
    return factor
