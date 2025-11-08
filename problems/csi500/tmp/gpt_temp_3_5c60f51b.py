import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Acceleration Divergence with Multi-timeframe Momentum factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Momentum & Acceleration
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_20d'] = data['close'] / data['close'].shift(20) - 1
    
    # Price Acceleration
    data['price_accel_5d'] = data['momentum_5d'] - data['momentum_5d'].shift(5)
    data['price_accel_20d'] = data['momentum_20d'] - data['momentum_20d'].shift(20)
    
    # Volume Momentum & Acceleration
    data['vol_momentum_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['vol_accel_5d'] = data['vol_momentum_5d'] - data['vol_momentum_5d'].shift(5)
    
    # Acceleration Divergence
    data['positive_div'] = ((data['price_accel_5d'] > 0) & (data['vol_accel_5d'] < 0)).astype(int)
    data['negative_div'] = ((data['price_accel_5d'] < 0) & (data['vol_accel_5d'] > 0)).astype(int)
    data['confirmed_move'] = (data['price_accel_5d'] * data['vol_accel_5d'] > 0).astype(int)
    
    # Volatility Context
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['vol_20d'] = data['close'].pct_change().rolling(window=20).std()
    data['vol_60d_percentile'] = data['vol_20d'].rolling(window=60).apply(
        lambda x: (x.iloc[-1] > np.percentile(x[:-1], 75)) if len(x) > 1 else 0
    )
    
    # Signal Strength Components
    # Acceleration strength
    data['accel_strength'] = np.abs(data['price_accel_5d']) * np.sign(data['price_accel_5d'])
    
    # Momentum alignment (5d and 20d momentum in same direction)
    data['momentum_align'] = ((data['momentum_5d'] > 0) & (data['momentum_20d'] > 0)).astype(int) - \
                            ((data['momentum_5d'] < 0) & (data['momentum_20d'] < 0)).astype(int)
    
    # Volume quality (consistent volume pattern)
    data['volume_quality'] = (data['volume'].rolling(window=10).std() / 
                             data['volume'].rolling(window=10).mean())
    
    # Composite Factor Calculation
    factor = np.zeros(len(data))
    
    for i in range(len(data)):
        if i < 60:  # Need enough data for volatility regime calculation
            factor[i] = 0
            continue
            
        # Strong Bullish: positive_div + high_accel + momentum_align + low_vol
        if (data['positive_div'].iloc[i] and 
            data['accel_strength'].iloc[i] > data['accel_strength'].rolling(window=20).mean().iloc[i] and
            data['momentum_align'].iloc[i] > 0 and
            not data['vol_60d_percentile'].iloc[i]):
            factor[i] = 2.0
            
        # Strong Bearish: negative_div + low_accel + momentum_misalign + low_vol
        elif (data['negative_div'].iloc[i] and 
              data['accel_strength'].iloc[i] < -data['accel_strength'].rolling(window=20).mean().iloc[i] and
              data['momentum_align'].iloc[i] < 0 and
              not data['vol_60d_percentile'].iloc[i]):
            factor[i] = -2.0
            
        # Moderate Bullish: positive_div + moderate_accel + momentum_align + normal_vol
        elif (data['positive_div'].iloc[i] and 
              data['accel_strength'].iloc[i] > 0 and
              data['momentum_align'].iloc[i] > 0 and
              data['vol_60d_percentile'].iloc[i]):
            factor[i] = 1.0
            
        # Moderate Bearish: negative_div + moderate_accel + momentum_misalign + normal_vol
        elif (data['negative_div'].iloc[i] and 
              data['accel_strength'].iloc[i] < 0 and
              data['momentum_align'].iloc[i] < 0 and
              data['vol_60d_percentile'].iloc[i]):
            factor[i] = -1.0
            
        # Weak: confirmed_move + high_vol + poor_volume_quality
        elif (data['confirmed_move'].iloc[i] and 
              data['vol_60d_percentile'].iloc[i] and
              data['volume_quality'].iloc[i] > data['volume_quality'].rolling(window=20).mean().iloc[i]):
            factor[i] = 0.5 * np.sign(data['price_accel_5d'].iloc[i])
        else:
            factor[i] = 0
    
    # Create output series
    factor_series = pd.Series(factor, index=data.index, name='factor')
    
    return factor_series
