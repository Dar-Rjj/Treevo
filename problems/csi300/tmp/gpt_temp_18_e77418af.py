import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate 5-day VWAP
    data['vwap_5'] = (data['close'] * data['volume']).rolling(window=5).sum() / data['volume'].rolling(window=5).sum()
    
    # Calculate VWAP slope using linear regression
    def calc_vwap_slope(series):
        if len(series) < 5:
            return np.nan
        x = np.arange(len(series))
        slope, _, _, _, _ = stats.linregress(x, series.values)
        return slope
    
    data['vwap_slope'] = data['vwap_5'].rolling(window=5).apply(calc_vwap_slope, raw=False)
    
    # Calculate 20-day price momentum
    data['price_momentum_20d'] = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)
    
    # Calculate momentum persistence (consecutive same-sign 5-day returns)
    data['ret_5d'] = data['close'].pct_change(5)
    data['ret_sign'] = np.sign(data['ret_5d'])
    data['momentum_persistence'] = 0
    
    for i in range(1, len(data)):
        if data['ret_sign'].iloc[i] == data['ret_sign'].iloc[i-1]:
            data.loc[data.index[i], 'momentum_persistence'] = data['momentum_persistence'].iloc[i-1] + 1
        else:
            data.loc[data.index[i], 'momentum_persistence'] = 1
    
    # Calculate 5-day volume change
    data['volume_change_5d'] = (data['volume'] - data['volume'].shift(5)) / data['volume'].shift(5)
    
    # Volume-price divergence analysis
    data['volume_confirmation'] = np.where(
        np.sign(data['vwap_slope']) == np.sign(data['volume_change_5d']), 1, -1
    )
    
    # Calculate divergence strength
    data['divergence_strength'] = data['vwap_slope'] * data['volume_change_5d']
    
    # Calculate convergence/divergence ratio
    data['convergence_ratio'] = data['divergence_strength'] / (data['price_momentum_20d'] + 1e-8)
    
    # Calculate true range efficiency
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    data['range_efficiency'] = abs(data['close'] - data['prev_close']) / data['true_range']
    
    # Generate composite alpha signal
    # Weight short-term VWAP momentum by volume confirmation
    vwap_weighted = data['vwap_slope'] * data['volume_confirmation']
    
    # Scale medium-term momentum by persistence score
    momentum_weighted = data['price_momentum_20d'] * (1 + data['momentum_persistence'] / 10)
    
    # Apply divergence multiplier
    divergence_multiplier = 1 + np.tanh(data['convergence_ratio'])
    
    # Combine trend components
    trend_component = (vwap_weighted * 0.6 + momentum_weighted * 0.4) * divergence_multiplier
    
    # Incorporate range efficiency
    final_signal = trend_component * data['range_efficiency']
    
    return final_signal
