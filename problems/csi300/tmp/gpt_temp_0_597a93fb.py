import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Components
    data['momentum_short'] = data['close'] / data['close'].shift(1) - 1
    data['momentum_medium'] = data['close'] / data['close'].shift(5) - 1
    
    # Turnover Components
    data['turnover_3d'] = (
        data['volume'] * data['close'] + 
        data['volume'].shift(1) * data['close'].shift(1) + 
        data['volume'].shift(2) * data['close'].shift(2)
    ) / 3
    
    data['turnover_5d'] = (
        data['volume'] * data['close'] + 
        data['volume'].shift(1) * data['close'].shift(1) + 
        data['volume'].shift(2) * data['close'].shift(2) + 
        data['volume'].shift(3) * data['close'].shift(3) + 
        data['volume'].shift(4) * data['close'].shift(4)
    ) / 5
    
    # Fractal Divergence
    data['fractal_divergence'] = (
        (data['momentum_short'] / data['momentum_medium']) * 
        (data['turnover_3d'] / data['turnover_5d']) * 
        (data['volume'] / data['volume'].shift(1))
    )
    
    # Volatility Components
    data['upside_vol'] = (data['high'] - data['open']) / data['open']
    data['downside_vol'] = (data['open'] - data['low']) / data['open']
    
    # Rejection Components
    data['upper_rejection'] = (data['high'] - data['close']) / (data['high'] - data['low'])
    data['lower_rejection'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    
    # Volatility Asymmetry
    data['volatility_asymmetry'] = (
        (data['upside_vol'] / data['downside_vol']) * 
        (data['upper_rejection'] - data['lower_rejection']) * 
        abs(data['close'] / data['close'].shift(1) - 1)
    )
    
    # Volume Components
    data['volume_ratio'] = data['volume'] / data['volume'].shift(1)
    
    # Calculate up-day volume concentration
    returns_5d = data['close'].pct_change()
    up_days_mask = returns_5d > 0
    data['up_day_volume_conc'] = (
        data['volume'].rolling(window=5).apply(
            lambda x: x[up_days_mask.loc[x.index]].sum() if up_days_mask.loc[x.index].any() else 0
        ) / data['volume'].rolling(window=5).sum()
    )
    
    # Price Components
    # Return asymmetry
    def calc_return_asymmetry(window):
        pos_returns = window[window > 0].sum()
        neg_returns = abs(window[window < 0]).sum()
        return np.log(1 + pos_returns) - np.log(1 + neg_returns)
    
    data['return_asymmetry'] = returns_5d.rolling(window=5).apply(calc_return_asymmetry, raw=False)
    
    # Range compression
    data['range_compression'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    
    # Volume-Price Flow
    data['volume_price_flow'] = (
        data['up_day_volume_conc'] * 
        data['return_asymmetry'] * 
        data['volume_ratio'] / data['range_compression']
    )
    
    # Final Alpha Synthesis
    data['base_divergence'] = data['fractal_divergence'] * data['volatility_asymmetry']
    data['volume_adjustment'] = data['volume_price_flow'] * data['volume_ratio']
    
    # Final Alpha Factor
    data['alpha_factor'] = (
        data['base_divergence'] * 
        data['volume_adjustment'] * 
        abs(data['close'] / data['close'].shift(1) - 1)
    )
    
    return data['alpha_factor']
