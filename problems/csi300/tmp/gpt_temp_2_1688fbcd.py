import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Efficiency Acceleration with Volume-Amount Divergence factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Efficiency Analysis
    # Short-term efficiency acceleration
    data['eff_2d'] = (data['close'] - data['close'].shift(2)) / (data['high'].rolling(3).max() - data['low'].rolling(3).min())
    data['eff_1d'] = (data['close'] - data['close'].shift(1)) / (data['high'].rolling(2).max() - data['low'].rolling(2).min())
    data['eff_accel_short'] = data['eff_2d'] - data['eff_1d']
    
    # Medium-term efficiency decay
    data['eff_5d'] = (data['close'] - data['close'].shift(5)) / (data['high'].rolling(6).max() - data['low'].rolling(6).min())
    data['eff_decay_med'] = data['eff_5d'] - data['eff_2d']
    
    # Long-term efficiency context
    data['eff_13d'] = (data['close'] - data['close'].shift(13)) / (data['high'].rolling(14).max() - data['low'].rolling(14).min())
    data['eff_decay_long'] = data['eff_13d'] - data['eff_5d']
    
    # Volume-Amount Divergence Analysis
    # Volume efficiency patterns
    data['vol_eff'] = data['volume'] / (data['high'] - data['low'])
    data['vol_eff_momentum'] = data['vol_eff'] - data['vol_eff'].shift(1)
    
    # Amount efficiency divergence
    data['amt_eff'] = data['amount'] / (data['high'] - data['low'])
    data['amt_vol_div'] = data['amt_eff'] - data['vol_eff']
    
    # Range-Adjusted Efficiency Analysis
    data['range_eff'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['range_eff_accel'] = data['range_eff'] - data['range_eff'].shift(1)
    
    # Efficiency Persistence Analysis
    # Acceleration streak analysis
    data['eff_accel_pos'] = (data['eff_accel_short'] > 0).astype(int)
    data['accel_streak'] = data['eff_accel_pos'].groupby(data.index).expanding().apply(lambda x: (x == 1).sum(), raw=False).reset_index(level=0, drop=True)
    
    # Cross-efficiency persistence
    data['vol_eff_pos'] = (data['vol_eff_momentum'] > 0).astype(int)
    data['cross_persistence'] = (data['eff_accel_pos'] == data['vol_eff_pos']).astype(int)
    
    # Donchian Efficiency Context
    data['eff_20d_high'] = data['eff_1d'].rolling(20).max()
    data['eff_20d_low'] = data['eff_1d'].rolling(20).min()
    data['eff_donchian_pos'] = (data['eff_1d'] - data['eff_20d_low']) / (data['eff_20d_high'] - data['eff_20d_low'])
    
    # Composite Acceleration Factor
    # Strong acceleration regime components
    data['multi_accel_convergence'] = (
        np.sign(data['eff_accel_short']) + 
        np.sign(data['eff_decay_med']) + 
        np.sign(data['eff_decay_long'])
    ) / 3.0
    
    # Volume-amount divergence confirmation
    data['div_strength'] = data['amt_vol_div'] * np.sign(data['eff_accel_short'])
    
    # Range expansion with efficiency improvement
    data['range_expansion'] = ((data['high'] - data['low']) > (data['high'].shift(1) - data['low'].shift(1))).astype(int)
    data['range_eff_signal'] = data['range_expansion'] * np.sign(data['range_eff_accel'])
    
    # Composite factor calculation
    data['composite_accel_factor'] = (
        data['multi_accel_convergence'] * 0.4 +
        data['div_strength'].fillna(0) * 0.3 +
        data['eff_donchian_pos'].fillna(0) * 0.2 +
        data['range_eff_signal'] * 0.1
    ) * data['accel_streak'].fillna(0)
    
    # Final factor with smoothing
    factor = data['composite_accel_factor'].rolling(3, min_periods=1).mean()
    
    return factor
