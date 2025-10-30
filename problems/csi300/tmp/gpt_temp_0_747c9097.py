import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Volatility-Regime Volume Acceleration Mean Reversion factor
    """
    df = data.copy()
    
    # Calculate daily returns
    df['returns'] = df['close'].pct_change()
    
    # Compute rolling volatility (20-day)
    df['volatility_20d'] = df['returns'].rolling(window=20).std()
    
    # Calculate 60-day median volatility
    df['volatility_median_60d'] = df['volatility_20d'].rolling(window=60).median()
    
    # Classify volatility regimes
    df['vol_regime'] = 'normal'
    high_vol_condition = df['volatility_20d'] > (1.5 * df['volatility_median_60d'])
    low_vol_condition = df['volatility_20d'] < (0.5 * df['volatility_median_60d'])
    df.loc[high_vol_condition, 'vol_regime'] = 'high'
    df.loc[low_vol_condition, 'vol_regime'] = 'low'
    
    # Calculate volume acceleration
    df['volume_ma_5d'] = df['volume'].rolling(window=5).mean()
    df['volume_ma_20d'] = df['volume'].rolling(window=20).mean()
    df['volume_acceleration'] = df['volume_ma_5d'] / df['volume_ma_20d']
    
    # Calculate additional components
    df['price_momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['normalized_range'] = (df['high'] - df['low']) / df['close']
    
    # Apply regime-specific factors
    factor_values = []
    
    for i in range(len(df)):
        if pd.isna(df.iloc[i]['vol_regime']):
            factor_values.append(np.nan)
            continue
            
        regime = df.iloc[i]['vol_regime']
        vol_20d = df.iloc[i]['volatility_20d']
        vol_acc = df.iloc[i]['volume_acceleration']
        
        if regime == 'high':
            # Strong mean reversion signal when volume acceleration > 1.2
            if vol_acc > 1.2:
                factor = -1 * (vol_20d * vol_acc)
            else:
                factor = -1 * vol_20d
                
        elif regime == 'normal':
            # Moderate momentum signal
            price_mom = df.iloc[i]['price_momentum_5d']
            factor = vol_acc * price_mom
            
        else:  # low volatility regime
            # Weak signal, focus on breakout detection
            norm_range = df.iloc[i]['normalized_range']
            factor = vol_acc * norm_range
            
        factor_values.append(factor)
    
    # Create output series
    factor_series = pd.Series(factor_values, index=df.index, name='vol_regime_volume_acc_factor')
    
    return factor_series
