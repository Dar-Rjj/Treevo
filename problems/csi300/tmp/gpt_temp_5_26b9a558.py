import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Momentum Components
    data['mom_1d'] = data['close'] / data['close'].shift(1) - 1
    data['mom_3d'] = data['close'] / data['close'].shift(3) - 1
    data['mom_5d'] = data['close'] / data['close'].shift(5) - 1
    
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
    data['fractal_div'] = (
        (data['mom_1d'] / data['mom_3d']) * 
        (data['volume'] / data['volume'].shift(1)) * 
        (data['mom_5d'] - data['mom_3d']) * 
        ((data['turnover_3d'] / data['turnover_5d']) - 1)
    )
    
    # Volatility Components
    data['upside_vol'] = (data['high'] - data['open']) / data['open']
    data['downside_vol'] = (data['open'] - data['low']) / data['open']
    
    # Rejection Components
    data['upper_rej'] = (data['high'] - data['close']) / (data['high'] - data['low'])
    data['lower_rej'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    
    # Asymmetry Flow
    data['asymmetry_flow'] = (
        (data['upside_vol'] / data['downside_vol']) / 
        (data['upside_vol'].shift(1) / data['downside_vol'].shift(1)) * 
        abs((data['close'] - data['close'].shift(1)) / data['close'].shift(1)) / 
        data['amount'] * 
        (data['upper_rej'] - data['lower_rej']) * 
        (data['volume'] / data['volume'].shift(1))
    )
    
    # Volume-Price Components
    returns = data['close'] / data['close'].shift(1) - 1
    up_days = returns > 0
    data['up_vol_ratio'] = (
        data['volume'].rolling(window=5).apply(lambda x: x[up_days.loc[x.index]].mean() if up_days.loc[x.index].any() else 0) / 
        data['volume'].rolling(window=5).mean()
    )
    
    pos_returns = returns.rolling(window=5).apply(lambda x: x[x > 0].sum())
    neg_returns = returns.rolling(window=5).apply(lambda x: abs(x[x < 0]).sum())
    data['price_asymmetry'] = np.log(1 + pos_returns) - np.log(1 + neg_returns)
    
    # Flow Components
    data['flow_imbalance'] = (data['amount'] - data['amount'].shift(1)) / data['amount'].shift(1)
    data['range_compression'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    
    # Flow Coherence
    data['flow_coherence'] = (
        data['up_vol_ratio'] * 
        data['price_asymmetry'] * 
        data['flow_imbalance'] * 
        abs(data['flow_imbalance']) / 
        data['range_compression'] * 
        (data['close'] / data['close'].shift(1) - 1)
    )
    
    # Core Components
    data['base_divergence'] = data['fractal_div'] * data['asymmetry_flow']
    data['volume_adjustment'] = (
        data['up_vol_ratio'] * 
        data['price_asymmetry'] * 
        (data['volume'] / data['volume'].shift(1))
    )
    
    # Regime Components
    data['high_vol_factor'] = (
        data['base_divergence'] * 
        abs((data['close'] - data['close'].shift(1)) / data['close'].shift(1)) / 
        data['amount'] * 
        (data['volume'] / data['volume'].shift(1))
    )
    
    data['low_vol_factor'] = (
        abs(data['flow_imbalance']) / 
        data['range_compression'] * 
        abs(abs((data['close'] - data['close'].shift(1)) / data['close'].shift(1)) / data['amount'] - 1) * 
        data['flow_coherence']
    )
    
    data['transition_factor'] = (
        abs((data['close'] - data['close'].shift(1)) / data['close'].shift(1)) - 
        abs(data['flow_imbalance']) * 
        ((data['upside_vol'] / data['downside_vol']) / (data['upside_vol'].shift(1) / data['downside_vol'].shift(1)) - 
         abs(data['flow_imbalance']) / data['range_compression']) * 
        (data['volume'] / data['volume'].shift(1))
    )
    
    # Final Alpha
    data['final_factor'] = (
        data['base_divergence'] * 
        data['volume_adjustment'] * 
        (data['high_vol_factor'] + data['low_vol_factor'] + data['transition_factor']) * 
        data['flow_coherence'] * 
        (data['volume'] / data['volume'].shift(1))
    )
    
    return data['final_factor']
