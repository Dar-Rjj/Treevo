import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily range percentage
    data['range_pct'] = (data['high'] - data['low']) / data['close'].shift(1)
    
    # Calculate rolling range volatility (10-day standard deviation)
    data['range_volatility'] = data['range_pct'].rolling(window=10, min_periods=5).std()
    
    # Identify volatility regimes
    volatility_threshold = data['range_volatility'].rolling(window=20, min_periods=10).median()
    data['regime'] = np.where(data['range_volatility'] > volatility_threshold * 1.2, 'high',
                     np.where(data['range_volatility'] < volatility_threshold * 0.8, 'low', 'transition'))
    
    # Intraday Efficiency Scoring
    # Price efficiency
    data['price_efficiency'] = np.abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['price_efficiency'] = np.where(data['high'] == data['low'], 0, data['price_efficiency'])
    
    # Volume efficiency
    data['volume_efficiency'] = data['volume'] / (data['high'] - data['low'])
    data['volume_efficiency'] = np.where(data['high'] == data['low'], 0, data['volume_efficiency'])
    
    # Normalize volume efficiency
    data['volume_efficiency_norm'] = data['volume_efficiency'] / data['volume_efficiency'].rolling(window=20, min_periods=10).mean()
    
    # Combined efficiency
    data['combined_efficiency'] = data['price_efficiency'] * data['volume_efficiency_norm']
    
    # Momentum Component
    # Short-term momentum (3-day)
    data['momentum_short'] = -(data['close'] / data['close'].shift(3) - 1)
    # Medium-term momentum (10-day)
    data['momentum_medium'] = -(data['close'] / data['close'].shift(10) - 1)
    # Momentum blend
    data['momentum_blend'] = data['momentum_short'] * data['momentum_medium']
    
    # Regime-Adaptive Signal Generation
    factor_values = []
    
    for i in range(len(data)):
        if i < 10:  # Need sufficient history
            factor_values.append(np.nan)
            continue
            
        current = data.iloc[i]
        regime = current['regime']
        
        if regime == 'high':
            # High volatility regime factor
            range_ratio = current['range_pct'] / data.iloc[i-10]['range_pct'] if data.iloc[i-10]['range_pct'] != 0 else 1
            factor = current['momentum_blend'] * (1 - current['combined_efficiency']) * range_ratio
            
        elif regime == 'low':
            # Low volatility regime factor
            volume_ratio = current['volume'] / data.iloc[i-10]['volume'] if data.iloc[i-10]['volume'] != 0 else 1
            factor = current['momentum_blend'] * current['combined_efficiency'] * volume_ratio
            
        else:  # Transition regime
            # Transition regime factor
            range_ratio = current['range_pct'] / data.iloc[i-5]['range_pct'] if data.iloc[i-5]['range_pct'] != 0 else 1
            volume_ratio = current['volume'] / data.iloc[i-5]['volume'] if data.iloc[i-5]['volume'] != 0 else 1
            factor = current['momentum_blend'] * range_ratio * volume_ratio
        
        factor_values.append(factor)
    
    # Create output series
    factor_series = pd.Series(factor_values, index=data.index, name='regime_adaptive_efficiency_momentum')
    
    return factor_series
