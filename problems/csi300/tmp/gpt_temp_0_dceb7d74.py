import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Adaptive Momentum Convergence with Dynamic Volatility Scaling
    """
    df = data.copy()
    
    # Price Momentum Signals
    df['price_mom_1'] = df['close'] / df['close'].shift(1) - 1
    df['price_mom_3'] = df['close'] / df['close'].shift(3) - 1
    df['price_mom_10'] = df['close'] / df['close'].shift(10) - 1
    df['price_mom_20'] = df['close'] / df['close'].shift(20) - 1
    
    # Volume Momentum Signals
    df['volume_mom_1'] = df['volume'] / df['volume'].shift(1) - 1
    df['volume_mom_3'] = df['volume'] / df['volume'].shift(3) - 1
    df['volume_mom_10'] = df['volume'] / df['volume'].shift(10) - 1
    df['volume_mom_20'] = df['volume'] / df['volume'].shift(20) - 1
    
    # Dynamic Volatility Framework
    df['price_range'] = df['high'] - df['low']
    
    # Initialize EMA columns
    df['range_ema_5'] = df['price_range']
    df['range_ema_20'] = df['price_range']
    
    # Calculate EMA for range_ema_5
    alpha_5 = 0.33
    for i in range(1, len(df)):
        df.loc[df.index[i], 'range_ema_5'] = (alpha_5 * df.loc[df.index[i], 'price_range'] + 
                                            (1 - alpha_5) * df.loc[df.index[i-1], 'range_ema_5'])
    
    # Calculate EMA for range_ema_20
    alpha_20 = 0.1
    for i in range(1, len(df)):
        df.loc[df.index[i], 'range_ema_20'] = (alpha_20 * df.loc[df.index[i], 'price_range'] + 
                                             (1 - alpha_20) * df.loc[df.index[i-1], 'range_ema_20'])
    
    # Volatility Scaling
    df['vol_scale_short'] = 1 / (df['range_ema_5'] + 0.0001)
    df['vol_scale_long'] = 1 / (df['range_ema_20'] + 0.0001)
    
    # Scaled Momentum Convergence
    df['conv_ultra_short'] = (df['price_mom_1'] - df['volume_mom_1']) * df['vol_scale_short']
    df['conv_short'] = (df['price_mom_3'] - df['volume_mom_3']) * df['vol_scale_short']
    df['conv_medium'] = (df['price_mom_10'] - df['volume_mom_10']) * df['vol_scale_long']
    df['conv_long'] = (df['price_mom_20'] - df['volume_mom_20']) * df['vol_scale_long']
    
    # Convergence Strength Assessment
    df['strength_ultra_short'] = np.abs(df['conv_ultra_short'])
    df['strength_short'] = np.abs(df['conv_short'])
    df['strength_medium'] = np.abs(df['conv_medium'])
    df['strength_long'] = np.abs(df['conv_long'])
    
    # Adaptive Timeframe Weighting
    df['vol_ratio'] = df['range_ema_5'] / (df['range_ema_20'] + 0.0001)
    
    df['weight_ultra_short'] = np.exp(-2 * df['vol_ratio'])
    df['weight_short'] = np.exp(-df['vol_ratio'])
    df['weight_medium'] = np.exp(-np.abs(df['vol_ratio'] - 1))
    df['weight_long'] = np.exp(df['vol_ratio'] - 1)
    
    # Weighted Alpha Components
    df['component_ultra_short'] = df['conv_ultra_short'] * df['weight_ultra_short']
    df['component_short'] = df['conv_short'] * df['weight_short']
    df['component_medium'] = df['conv_medium'] * df['weight_medium']
    df['component_long'] = df['conv_long'] * df['weight_long']
    
    # Final Alpha Construction
    df['raw_alpha'] = (df['component_ultra_short'] + df['component_short'] + 
                      df['component_medium'] + df['component_long'])
    
    df['alpha_output'] = df['raw_alpha'] / (np.abs(df['raw_alpha']) + 0.0001)
    
    return df['alpha_output']
