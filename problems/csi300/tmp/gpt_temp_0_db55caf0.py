import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility Structure
    data['true_range_vol'] = (data['high'] - data['low']) / data['close']
    data['gap_vol'] = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['vol_ratio'] = data['true_range_vol'] / data['gap_vol']
    
    # Volume Dynamics
    data['volume_momentum'] = data['volume'] / data['volume'].shift(3) - 1
    data['volume_clustering'] = data['volume'] / ((data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3)) / 3)
    data['volume_fractal'] = (data['volume'] / data['volume'].shift(1)) - (data['volume'].shift(1) / data['volume'].shift(2))
    
    # Price-Volume Cointegration
    data['pv_divergence'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1) - data['volume_momentum']
    data['pv_acceleration'] = ((data['close'] - 2 * data['close'].shift(1) + data['close'].shift(2)) / data['close'].shift(2)) - \
                             ((data['volume'] - 2 * data['volume'].shift(1) + data['volume'].shift(2)) / data['volume'].shift(2))
    
    # Gap Integration
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_gap'] = (data['close'] - data['open']) / data['open']
    
    # Breakout Dynamics
    data['range_expansion'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    
    # Rolling highs and lows for breakout calculations
    data['high_rolling_max'] = data['high'].rolling(window=4, min_periods=1).apply(lambda x: x[:-1].max() if len(x) > 1 else np.nan)
    data['low_rolling_min'] = data['low'].rolling(window=4, min_periods=1).apply(lambda x: x[:-1].min() if len(x) > 1 else np.nan)
    
    data['upper_breakout'] = (data['high'] - data['high_rolling_max']) / data['close'].shift(1)
    data['lower_breakout'] = (data['low_rolling_min'] - data['low']) / data['close'].shift(1)
    data['pressure_imbalance'] = ((data['high'] - data['close']) - (data['close'] - data['low'])) / (data['high'] - data['low'])
    
    # Regime Detection
    data['vol_regime'] = data['true_range_vol'] > data['true_range_vol'].rolling(window=4, min_periods=1).mean()
    data['volume_regime'] = np.select([
        data['volume_clustering'] > 1.2,
        data['volume_clustering'] < 0.8
    ], ['clustered', 'dispersed'], default='normal')
    data['cointegration_regime'] = (data['pv_divergence'] * data['pv_acceleration']) > 0
    
    # Alpha Synthesis
    data['volume_avg_4'] = data['volume'].rolling(window=4, min_periods=1).mean()
    
    # High Vol Expansion
    high_vol_expansion = (np.sign(data['intraday_gap']) != np.sign(data['overnight_gap'])) * \
                        (data['volume'] / data['volume_avg_4']) * data['pv_acceleration']
    
    # Low Vol Contraction
    low_vol_contraction = (np.sign(data['intraday_gap']) == np.sign(data['overnight_gap'])) * \
                         data['range_expansion'] * data['pv_divergence']
    
    # Convergent Clustered
    convergent_clustered = data['pv_acceleration'] * data['volume_momentum'] * data['true_range_vol']
    
    # Divergent Dispersed
    divergent_dispersed = data['pv_divergence'] * data['volume_fractal'] * data['vol_ratio']
    
    # Composite Alpha with regime weighting
    alpha_composite = np.zeros(len(data))
    
    for i in range(len(data)):
        if data['vol_regime'].iloc[i] and data['volume_regime'].iloc[i] == 'clustered' and data['cointegration_regime'].iloc[i]:
            alpha_composite[i] = convergent_clustered.iloc[i]
        elif not data['vol_regime'].iloc[i] and data['volume_regime'].iloc[i] == 'dispersed' and not data['cointegration_regime'].iloc[i]:
            alpha_composite[i] = divergent_dispersed.iloc[i]
        elif data['vol_regime'].iloc[i]:
            alpha_composite[i] = high_vol_expansion.iloc[i]
        else:
            alpha_composite[i] = low_vol_contraction.iloc[i]
    
    return pd.Series(alpha_composite, index=data.index)
