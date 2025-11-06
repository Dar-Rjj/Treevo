import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Multi-Timeframe Volatility-Scaled Price-Volume Divergence alpha factor
    """
    df = data.copy()
    
    # Initialize EMA columns
    df['EMA_fast_p'] = df['close']
    df['EMA_med_p'] = df['close']
    df['EMA_slow_p'] = df['close']
    df['EMA_fast_v'] = df['volume']
    df['EMA_med_v'] = df['volume']
    df['EMA_slow_v'] = df['volume']
    df['EMA_range'] = df['high'] - df['low']
    df['EMA_vol_chg'] = abs(df['volume'].diff())
    
    # Fill NaN values for initial calculations
    df = df.fillna(method='bfill')
    
    # Calculate EMAs iteratively
    for i in range(1, len(df)):
        # Price EMAs
        df.loc[df.index[i], 'EMA_fast_p'] = 0.4 * df.loc[df.index[i], 'close'] + 0.6 * df.loc[df.index[i-1], 'EMA_fast_p']
        df.loc[df.index[i], 'EMA_med_p'] = 0.2 * df.loc[df.index[i], 'close'] + 0.8 * df.loc[df.index[i-1], 'EMA_med_p']
        df.loc[df.index[i], 'EMA_slow_p'] = 0.1 * df.loc[df.index[i], 'close'] + 0.9 * df.loc[df.index[i-1], 'EMA_slow_p']
        
        # Volume EMAs
        df.loc[df.index[i], 'EMA_fast_v'] = 0.4 * df.loc[df.index[i], 'volume'] + 0.6 * df.loc[df.index[i-1], 'EMA_fast_v']
        df.loc[df.index[i], 'EMA_med_v'] = 0.2 * df.loc[df.index[i], 'volume'] + 0.8 * df.loc[df.index[i-1], 'EMA_med_v']
        df.loc[df.index[i], 'EMA_slow_v'] = 0.1 * df.loc[df.index[i], 'volume'] + 0.9 * df.loc[df.index[i-1], 'EMA_slow_v']
        
        # Range and Volume Change EMAs
        df.loc[df.index[i], 'EMA_range'] = 0.2 * (df.loc[df.index[i], 'high'] - df.loc[df.index[i], 'low']) + 0.8 * df.loc[df.index[i-1], 'EMA_range']
        vol_chg = abs(df.loc[df.index[i], 'volume'] - df.loc[df.index[i-1], 'volume'])
        df.loc[df.index[i], 'EMA_vol_chg'] = 0.2 * vol_chg + 0.8 * df.loc[df.index[i-1], 'EMA_vol_chg']
    
    # Volatility Scaling
    df['price_vol_scale'] = 1 / (df['EMA_range'] + 0.0001)
    df['volume_vol_scale'] = 1 / (df['EMA_vol_chg'] + 0.0001)
    
    # Volatility-Scaled Momentum
    df['fast_price_mom'] = (df['EMA_fast_p'] - df['EMA_med_p']) * df['price_vol_scale']
    df['medium_price_mom'] = (df['EMA_med_p'] - df['EMA_slow_p']) * df['price_vol_scale']
    df['slow_price_mom'] = (df['EMA_fast_p'] - df['EMA_slow_p']) * df['price_vol_scale']
    
    df['fast_volume_mom'] = (df['EMA_fast_v'] - df['EMA_med_v']) * df['volume_vol_scale']
    df['medium_volume_mom'] = (df['EMA_med_v'] - df['EMA_slow_v']) * df['volume_vol_scale']
    df['slow_volume_mom'] = (df['EMA_fast_v'] - df['EMA_slow_v']) * df['volume_vol_scale']
    
    # Price-Volume Divergence
    df['fast_divergence'] = df['fast_price_mom'] - df['fast_volume_mom']
    df['medium_divergence'] = df['medium_price_mom'] - df['medium_volume_mom']
    df['slow_divergence'] = df['slow_price_mom'] - df['slow_volume_mom']
    
    # Volatility Regime Assessment
    df['volatility_ratio'] = df['EMA_range'] / (df['EMA_range'].shift(5) + 0.0001)
    
    # Continuous Weight Functions
    df['fast_weight'] = np.maximum(0, np.minimum(1, 2 - df['volatility_ratio']))
    df['medium_weight'] = np.maximum(0, np.minimum(1, 1 - abs(1 - df['volatility_ratio'])))
    df['slow_weight'] = np.maximum(0, np.minimum(1, df['volatility_ratio'] - 1))
    
    # Final Alpha Construction
    df['fast_component'] = df['fast_divergence'] * df['fast_weight']
    df['medium_component'] = df['medium_divergence'] * df['medium_weight']
    df['slow_component'] = df['slow_divergence'] * df['slow_weight']
    
    alpha = df['fast_component'] + df['medium_component'] + df['slow_component']
    
    return alpha
