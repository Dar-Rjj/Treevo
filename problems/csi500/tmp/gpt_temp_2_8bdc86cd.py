import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Asymmetric Momentum-Volume Regime Factor
    Combines multi-timeframe volatility-scaled momentum with volume regime detection
    """
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Asymmetric Momentum Framework
    # Daily Range for volatility adjustment
    data['daily_range'] = (data['high'] - data['low']) / data['close'].shift(1)
    
    # Short-term Momentum (2-day)
    data['m2_raw'] = data['close'] / data['close'].shift(2) - 1
    data['m2_scaled'] = data['m2_raw'] / (data['daily_range'] + 0.001)
    
    # Medium-term Momentum (7-day)
    data['m7_raw'] = data['close'] / data['close'].shift(7) - 1
    data['m7_scaled'] = data['m7_raw'] / (data['daily_range'] + 0.001)
    
    # Long-term Momentum (15-day)
    data['m15_raw'] = data['close'] / data['close'].shift(15) - 1
    data['m15_scaled'] = data['m15_raw'] / (data['daily_range'] + 0.001)
    
    # Volume Regime Detection
    # 20-day rolling volume percentile
    data['volume_pct'] = data['volume'].rolling(window=20).apply(
        lambda x: (x.rank(pct=True).iloc[-1] * 100) if len(x) == 20 else np.nan, 
        raw=False
    )
    
    # Nonlinear Volume Transformation (Cube Root)
    data['volume_cube'] = (data['volume_pct'] / 100) ** (1/3)
    
    # Volume Regime Classification
    def classify_volume(cube_val):
        if cube_val > 0.9:  # percentile > 72.9
            return 'high'
        elif cube_val >= 0.7:  # percentile 34.3-72.9
            return 'medium'
        else:  # percentile < 34.3
            return 'low'
    
    data['volume_regime'] = data['volume_cube'].apply(classify_volume)
    
    # Multiplicative Factor Construction
    # Momentum combination with cube root stabilization
    data['momentum_product'] = data['m2_scaled'] * data['m7_scaled'] * data['m15_scaled']
    data['momentum_combined'] = np.sign(data['momentum_product']) * (abs(data['momentum_product'])) ** (1/3)
    
    # Volume Regime Weighting
    def apply_volume_weight(row):
        momentum_val = row['momentum_combined']
        cube_val = row['volume_cube']
        
        if row['volume_regime'] == 'high':
            return momentum_val * (1 + cube_val)
        elif row['volume_regime'] == 'medium':
            return momentum_val * (1 + 0.5 * cube_val)
        else:  # low volume
            return momentum_val * (1 + 0.25 * cube_val)
    
    data['combined_factor'] = data.apply(apply_volume_weight, axis=1)
    
    # Directional Enhancement - amplify when momentum and volume trends are concordant
    momentum_direction = np.sign(data['momentum_combined'])
    volume_trend = data['volume_cube'].diff(5)  # 5-day volume trend
    
    # Amplify when both momentum and volume are trending in the same direction
    enhancement_mask = (momentum_direction * np.sign(volume_trend)) > 0
    data.loc[enhancement_mask, 'combined_factor'] = data.loc[enhancement_mask, 'combined_factor'] * 1.2
    
    # Final Factor Output with hyperbolic tangent
    factor = np.tanh(data['combined_factor'])
    
    return factor
