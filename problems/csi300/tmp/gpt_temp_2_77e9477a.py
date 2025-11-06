import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Scale Volatility Structure
    data['true_range_vol'] = (data['high'] - data['low']) / data['close']
    data['gap_vol'] = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['vol_ratio'] = data['true_range_vol'] / data['gap_vol'].replace(0, np.nan)
    
    data['vol_regime_short'] = data['close'].rolling(window=5).std()
    data['vol_regime_long'] = data['close'].rolling(window=10).std()
    data['multi_scale_vol_regime'] = data['vol_regime_short'] / data['vol_regime_long']
    
    # Fractal Volume Dynamics
    data['volume_momentum'] = data['volume'] / data['volume'].shift(3) - 1
    data['volume_clustering'] = data['volume'] / ((data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3)) / 3)
    
    data['volume_fractal'] = (data['volume'] / data['volume'].shift(1)) - (data['volume'].shift(1) / data['volume'].shift(2))
    
    # Volume Direction Consistency
    data['price_direction'] = np.sign(data['close'] - data['open'])
    data['direction_change'] = data['price_direction'] != data['price_direction'].shift(1)
    data['consistency_counter'] = data.groupby(data['direction_change'].cumsum()).cumcount() + 1
    data['volume_direction_consistency'] = data['consistency_counter']
    
    data['volume_acceleration'] = ((data['volume'] / data['volume'].shift(1) - 1) - 
                                 (data['volume'].shift(1) / data['volume'].shift(2) - 1))
    data['volume_flow_signal'] = (1 + data['volume_direction_consistency'] / 5) * (1 + data['volume_acceleration'])
    
    # Price-Volume Cointegration Framework
    data['price_momentum'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    data['pv_divergence'] = data['price_momentum'] - data['volume_momentum']
    
    data['pv_acceleration'] = ((data['close'] - 2 * data['close'].shift(1) + data['close'].shift(2)) / data['close'].shift(2) - 
                             (data['volume'] - 2 * data['volume'].shift(1) + data['volume'].shift(2)) / data['volume'].shift(2))
    
    data['pv_fractal_coherence'] = (np.sign(data['close'] - data['close'].shift(1)) * data['volume_fractal'] - 
                                  np.sign(data['close'].shift(1) - data['close'].shift(2)) * data['volume_fractal'].shift(1))
    
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_gap'] = (data['close'] - data['open']) / data['open']
    data['momentum_divergence'] = ((data['close'] - data['close'].shift(2)) / data['close'].shift(2) - 
                                 (data['close'] - data['close'].shift(6)) / data['close'].shift(6))
    
    # Fractal Breakout System
    data['current_range'] = data['high'] - data['low']
    data['prev_range'] = data['high'].shift(1) - data['low'].shift(1)
    data['range_expansion'] = data['current_range'] / data['prev_range']
    
    data['upper_breakout'] = (data['high'] - data['high'].rolling(window=4, min_periods=1).apply(lambda x: x[:-1].max() if len(x) > 1 else np.nan)) / data['close'].shift(1)
    data['lower_breakout'] = (data['low'].rolling(window=4, min_periods=1).apply(lambda x: x[:-1].min() if len(x) > 1 else np.nan) - data['low']) / data['close'].shift(1)
    data['breakout_direction'] = np.sign(data['close'] - data['open']) * np.sign(data['high'] - data['high'].shift(1))
    
    data['upper_pressure'] = (data['high'] - data['close']) / data['current_range']
    data['lower_pressure'] = (data['close'] - data['low']) / data['current_range']
    data['pressure_imbalance'] = data['upper_pressure'] - data['lower_pressure']
    
    # Multi-Regime Detection
    data['vol_regime'] = data['true_range_vol'] > data['true_range_vol'].rolling(window=4).mean()
    
    data['volume_regime'] = 'Normal'
    data.loc[data['volume_clustering'] > 1.2, 'volume_regime'] = 'Clustered'
    data.loc[data['volume_clustering'] < 0.8, 'volume_regime'] = 'Dispersed'
    
    data['cointegration_regime'] = 'Neutral'
    data.loc[data['pv_divergence'] * data['pv_acceleration'] > 0, 'cointegration_regime'] = 'Convergent'
    data.loc[data['pv_divergence'] * data['pv_acceleration'] < 0, 'cointegration_regime'] = 'Divergent'
    
    data['multi_scale_regime'] = 'Low Vol Contraction'
    data.loc[data['multi_scale_vol_regime'] > 1.2, 'multi_scale_regime'] = 'High Vol Expansion'
    
    # Adaptive Alpha Synthesis
    # High Vol Expansion Regime
    data['gap_reversal'] = np.sign(data['intraday_gap']) != np.sign(data['overnight_gap'])
    data['volume_spike'] = data['volume'] / data['volume'].rolling(window=4).mean()
    data['price_efficiency'] = abs(data['intraday_gap']) / (data['current_range'] / data['open'])
    data['high_vol_factor'] = data['gap_reversal'].astype(float) * data['volume_spike'] * data['price_efficiency'] * data['pv_fractal_coherence']
    
    # Low Vol Contraction Regime
    data['gap_persistence'] = np.sign(data['intraday_gap']) == np.sign(data['overnight_gap'])
    data['volume_trend_consistency'] = (np.sign(data['volume'] - data['volume'].shift(1)) == 
                                      np.sign(data['volume'].shift(1) - data['volume'].shift(2))).astype(float)
    data['range_breakout_strength'] = data['range_expansion'] * data['breakout_direction']
    data['low_vol_factor'] = data['gap_persistence'].astype(float) * data['volume_trend_consistency'] * data['range_breakout_strength'] * data['pv_divergence']
    
    # Convergent Clustered Regime
    data['convergent_factor'] = data['pv_acceleration'] * data['volume_momentum'] * data['true_range_vol'] * data['volume_flow_signal']
    
    # Divergent Dispersed Regime
    data['divergent_factor'] = data['pv_divergence'] * data['volume_fractal'] * data['vol_ratio'] * data['momentum_divergence']
    
    # Composite Alpha Generation
    conditions = [
        (data['multi_scale_regime'] == 'High Vol Expansion') & 
        ((data['volume_regime'] == 'Clustered') | (data['cointegration_regime'] == 'Convergent')),
        
        (data['multi_scale_regime'] == 'Low Vol Contraction') & 
        ((data['volume_regime'] == 'Dispersed') | (data['cointegration_regime'] == 'Divergent')),
        
        (data['cointegration_regime'] == 'Convergent') & (data['volume_regime'] == 'Clustered'),
        
        (data['cointegration_regime'] == 'Divergent') & (data['volume_regime'] == 'Dispersed')
    ]
    
    choices = [
        data['high_vol_factor'],
        data['low_vol_factor'],
        data['convergent_factor'],
        data['divergent_factor']
    ]
    
    data['alpha'] = np.select(conditions, choices, 
                            default=(data['high_vol_factor'] + data['low_vol_factor'] + 
                                   data['convergent_factor'] + data['divergent_factor']) / 4)
    
    return data['alpha']
