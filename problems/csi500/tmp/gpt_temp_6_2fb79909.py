import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum with Volume Confirmation factor
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate momentum signals
    df['momentum_2d'] = df['close'] / df['close'].shift(2) - 1
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    
    # Calculate returns for volatility
    df['ret'] = df['close'] / df['close'].shift(1) - 1
    
    # Volatility regime classification
    df['vol_5d'] = df['ret'].rolling(window=5).std()
    df['vol_20d'] = df['ret'].rolling(window=20).std()
    
    # Regime detection
    df['vol_regime'] = 'normal'
    high_vol_condition = df['vol_5d'] > (df['vol_20d'] * 1.2)
    low_vol_condition = df['vol_5d'] < (df['vol_20d'] * 0.8)
    df.loc[high_vol_condition, 'vol_regime'] = 'high'
    df.loc[low_vol_condition, 'vol_regime'] = 'low'
    
    # Volume dynamics
    df['volume_change_3d'] = df['volume'] / df['volume'].shift(3) - 1
    df['volume_acceleration'] = (df['volume'] / df['volume'].shift(3)) / (df['volume'].shift(3) / df['volume'].shift(6)) - 1
    
    # Price-volume correlation
    df['volume_change'] = df['volume'] / df['volume'].shift(1) - 1
    
    # Calculate rolling correlation
    corr_values = []
    for i in range(len(df)):
        if i >= 5:
            price_returns = df['ret'].iloc[i-4:i+1].values
            volume_changes = df['volume_change'].iloc[i-4:i+1].values
            if len(price_returns) == 5 and len(volume_changes) == 5:
                corr = np.corrcoef(price_returns, volume_changes)[0, 1]
                corr_values.append(corr if not np.isnan(corr) else 0)
            else:
                corr_values.append(0)
        else:
            corr_values.append(0)
    
    df['price_volume_corr'] = corr_values
    
    # Volume confirmation strength
    df['volume_confirmation'] = 'neutral'
    
    strong_conf_cond = (df['volume_acceleration'] > 0) & (df['price_volume_corr'] > 0.1)
    weak_conf_cond = (df['volume_acceleration'] > 0) | (df['price_volume_corr'] > 0.1)
    strong_div_cond = (df['volume_acceleration'] < -0.1) & (df['price_volume_corr'] > 0.1)
    weak_div_cond = (df['volume_acceleration'] < 0) | (df['price_volume_corr'] < -0.1)
    
    df.loc[strong_conf_cond, 'volume_confirmation'] = 'strong_confirmation'
    df.loc[weak_conf_cond & ~strong_conf_cond, 'volume_confirmation'] = 'weak_confirmation'
    df.loc[strong_div_cond, 'volume_confirmation'] = 'strong_divergence'
    df.loc[weak_div_cond & ~strong_div_cond, 'volume_confirmation'] = 'weak_divergence'
    
    # Adaptive factor integration
    for i in range(len(df)):
        if i < 20:  # Need enough data for calculations
            result.iloc[i] = 0
            continue
            
        # Regime-based momentum selection
        regime = df['vol_regime'].iloc[i]
        if regime == 'high':
            selected_momentum = df['momentum_2d'].iloc[i]
        elif regime == 'low':
            selected_momentum = df['momentum_10d'].iloc[i]
        else:  # normal
            selected_momentum = df['momentum_5d'].iloc[i]
        
        # Volume confirmation multiplier
        confirmation = df['volume_confirmation'].iloc[i]
        if confirmation == 'strong_confirmation':
            multiplier = 1.3
        elif confirmation == 'weak_confirmation':
            multiplier = 1.1
        elif confirmation == 'strong_divergence':
            multiplier = 0.5
        elif confirmation == 'weak_divergence':
            multiplier = 0.8
        else:  # neutral
            multiplier = 1.0
        
        # Correlation direction adjustment
        correlation = df['price_volume_corr'].iloc[i]
        if correlation < 0:
            selected_momentum = -selected_momentum
        
        # Final factor value
        result.iloc[i] = selected_momentum * multiplier
    
    return result
