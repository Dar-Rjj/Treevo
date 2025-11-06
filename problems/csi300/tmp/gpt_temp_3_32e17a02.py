import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility Regime Classification
    # Short-term volatility
    data['short_term_vol'] = (data['high'] - data['low']) / data['close']
    
    # Medium-term volatility
    data['medium_term_vol'] = data['close'].rolling(window=10).std() / data['close']
    
    # Combined volatility measure
    data['combined_vol'] = (data['short_term_vol'] + data['medium_term_vol']) / 2
    
    # Regime threshold using rolling percentile
    data['vol_percentile'] = data['combined_vol'].rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Volatility regime classification
    data['vol_regime'] = np.where(data['vol_percentile'] < 0.33, 'low',
                         np.where(data['vol_percentile'] > 0.67, 'high', 'transition'))
    
    # Regime-Specific Factor Construction
    
    # Low volatility regime factors
    data['ma_5'] = data['close'].rolling(window=5).mean()
    data['mean_reversion'] = (data['close'] - data['ma_5']) / (data['high'] - data['low'])
    
    data['volume_ma_5'] = data['volume'].rolling(window=5).mean()
    data['price_change_dir'] = np.where(data['close'] > data['close'].shift(1), 1, -1)
    data['volume_confirmation'] = (data['volume'] / data['volume_ma_5']) * data['price_change_dir']
    
    # High volatility regime factors
    data['momentum_persistence'] = ((data['close'] / data['close'].shift(1) - 1) * 
                                   (data['close'].shift(1) / data['close'].shift(2) - 1))
    
    data['vol_breakout'] = ((data['high'] - data['high'].shift(1)) / 
                           (data['high'].shift(1) - data['low'].shift(1)))
    data['vol_breakout'] = data['vol_breakout'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Transition regime factors
    data['regime_change'] = data['vol_regime'] != data['vol_regime'].shift(1)
    data['distance_from_boundary'] = np.abs(data['vol_percentile'] - 0.5) * 2
    
    # Signal Integration
    factors = []
    
    for i in range(len(data)):
        if i < 20:  # Ensure enough data for calculations
            factors.append(0)
            continue
            
        current_regime = data['vol_regime'].iloc[i]
        
        if current_regime == 'low':
            # Low volatility regime: emphasize mean reversion and volume confirmation
            factor_value = (0.6 * data['mean_reversion'].iloc[i] + 
                          0.4 * data['volume_confirmation'].iloc[i])
            
        elif current_regime == 'high':
            # High volatility regime: emphasize momentum and volatility breakout
            factor_value = (0.5 * data['momentum_persistence'].iloc[i] + 
                          0.5 * data['vol_breakout'].iloc[i])
            
        else:  # Transition regime
            # Blend factors with adaptive weighting based on distance from boundaries
            boundary_weight = data['distance_from_boundary'].iloc[i]
            
            low_vol_factor = (0.6 * data['mean_reversion'].iloc[i] + 
                            0.4 * data['volume_confirmation'].iloc[i])
            
            high_vol_factor = (0.5 * data['momentum_persistence'].iloc[i] + 
                             0.5 * data['vol_breakout'].iloc[i])
            
            # Weight based on proximity to regime boundaries
            if data['vol_percentile'].iloc[i] < 0.5:
                # Closer to low volatility
                factor_value = (boundary_weight * low_vol_factor + 
                              (1 - boundary_weight) * high_vol_factor)
            else:
                # Closer to high volatility
                factor_value = (boundary_weight * high_vol_factor + 
                              (1 - boundary_weight) * low_vol_factor)
            
            # Add regime change alert
            if data['regime_change'].iloc[i]:
                factor_value *= 1.2  # Amplify signal during regime transitions
        
        factors.append(factor_value)
    
    # Create output series
    factor_series = pd.Series(factors, index=data.index, name='volatility_regime_adaptive_alpha')
    
    # Handle any remaining NaN values
    factor_series = factor_series.fillna(0)
    
    return factor_series
