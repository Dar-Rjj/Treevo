import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Fractal Amplitude Ratio
    # 3-day fractal amplitude
    data['range_3d_avg'] = ((data['high'] - data['low']) + 
                           (data['high'].shift(1) - data['low'].shift(1)) + 
                           (data['high'].shift(2) - data['low'].shift(2))) / 3
    data['fractal_amp_3d'] = ((data['high'] - data['low']) / data['range_3d_avg']) - 1
    
    # 10-day fractal amplitude
    data['range_10d_avg'] = data['high'].rolling(window=10).apply(lambda x: (x - data.loc[x.index, 'low']).mean(), raw=False)
    data['fractal_amp_10d'] = (data['range_3d_avg'] / data['range_10d_avg']) - 1
    
    # Fractal amplitude ratio
    data['fractal_amp_ratio'] = data['fractal_amp_3d'] / data['fractal_amp_10d']
    
    # Volatility Ratio
    data['atr_3d'] = (data['true_range'] + data['true_range'].shift(1) + data['true_range'].shift(2)) / 3
    data['atr_8d'] = data['true_range'].rolling(window=8).mean()
    data['volatility_ratio'] = data['atr_3d'] / data['atr_8d']
    
    # Momentum Divergence
    data['momentum_5d'] = (data['close'] / data['close'].shift(5)) - 1
    data['momentum_10d'] = (data['close'] / data['close'].shift(10)) - 1
    data['momentum_divergence'] = data['momentum_5d'] - data['momentum_10d']
    
    # Volume-Amount Alignment
    # Volume transition score
    data['volume_ratio'] = (data['volume'] / data['volume'].shift(3)) - 1
    data['volume_var_8d'] = data['volume'].rolling(window=8).var()
    data['volume_transition_score'] = data['volume_ratio'] / data['volume_var_8d']
    
    # Amount fractal momentum
    data['amount_momentum'] = (data['amount'] / data['amount'].shift(3)) - 1
    
    # Signal Construction
    data['volume_enhanced_divergence'] = data['momentum_divergence'] * data['volume_transition_score']
    data['volume_amount_alignment'] = data['volume_transition_score'] * data['amount_momentum']
    
    # Elasticity Weighting
    conditions = [
        (data['fractal_amp_ratio'] > 0.3) | (data['volatility_ratio'] > 1.2),
        (data['fractal_amp_ratio'] < -0.2) | (data['volatility_ratio'] < 0.8)
    ]
    choices = [
        0.6 * data['volume_enhanced_divergence'] + 0.4 * data['volume_amount_alignment'],
        0.3 * data['volume_enhanced_divergence'] + 0.7 * data['volume_amount_alignment']
    ]
    data['weighted_signal'] = np.select(conditions, choices, 
                                       0.5 * data['volume_enhanced_divergence'] + 0.5 * data['volume_amount_alignment'])
    
    # Volatility-dampened signal
    data['volatility_dampened_signal'] = (data['weighted_signal'] * (2 - data['volatility_ratio']) / 
                                         (1 + abs(data['fractal_amp_ratio'])))
    
    # Combined volatility
    data['range_vol_20d'] = ((data['high'] - data['low']).rolling(window=20).apply(lambda x: np.sqrt((x**2).mean()), raw=False))
    data['return_vol_20d'] = (data['close'].pct_change().rolling(window=20).apply(lambda x: np.sqrt((x**2).mean()), raw=False))
    data['combined_volatility'] = data['range_vol_20d'] * data['return_vol_20d']
    
    # Final Alpha Factor
    alpha = data['volatility_dampened_signal'] / (1 + abs(data['combined_volatility']))
    
    # Clean up intermediate columns
    result = alpha.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return result
