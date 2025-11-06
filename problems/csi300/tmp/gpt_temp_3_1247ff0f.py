import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Regime Transition Detection
    data['volume_regime_shift'] = ((data['volume'] - data['volume'].shift(3)) / (data['volume'].shift(3) + 0.001)) * np.sign(data['volume'] - data['volume'].shift(1))
    data['volatility_regime_shift'] = ((data['high'] - data['low']) / (data['high'].shift(3) - data['low'].shift(3) + 0.001)) * np.sign((data['high'] - data['low']) - (data['high'].shift(1) - data['low'].shift(1)))
    data['price_regime_shift'] = ((data['close'] - data['close'].shift(3)) / (data['close'].shift(3) + 0.001)) * np.sign(data['close'] - data['close'].shift(1))
    data['regime_transition_score'] = data['volume_regime_shift'] * data['volatility_regime_shift'] * data['price_regime_shift'] * np.sign(data['volume_regime_shift'] - data['price_regime_shift'])
    
    # Momentum Acceleration Patterns
    data['price_acceleration'] = ((data['close'] - data['close'].shift(2)) / (data['close'].shift(2) - data['close'].shift(4) + 0.001)) * ((data['high'] - data['low']) / (data['high'].shift(2) - data['low'].shift(2) + 0.001))
    data['volume_acceleration'] = ((data['volume'] - data['volume'].shift(2)) / (data['volume'].shift(2) - data['volume'].shift(4) + 0.001)) * np.sign(data['volume'] - data['volume'].shift(1))
    data['volatility_acceleration'] = ((data['high'] - data['low']) / (data['high'].shift(2) - data['low'].shift(2) + 0.001)) * np.sign((data['high'] - data['low']) - (data['high'].shift(1) - data['low'].shift(1)))
    data['acceleration_convergence'] = data['price_acceleration'] * data['volume_acceleration'] * data['volatility_acceleration'] * np.sign(data['price_acceleration'] - data['volatility_acceleration'])
    
    # Gap Momentum Dynamics
    data['opening_gap_momentum'] = ((data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + 0.001)) * (data['volume'] / (data['volume'].shift(1) + 0.001))
    data['intraday_gap_momentum'] = ((data['close'] - data['open']) / (data['open'] - data['close'].shift(1) + 0.001)) * ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 0.001))
    data['gap_momentum_ratio'] = data['opening_gap_momentum'] / (data['intraday_gap_momentum'] + 0.001) * np.sign(data['opening_gap_momentum'] - data['intraday_gap_momentum'])
    
    # Multi-Scale Momentum Alignment
    data['ultra_short_momentum'] = ((data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 0.001)) * (data['volume'] / (data['volume'].shift(1) + 0.001))
    
    # Rolling windows for short and medium scale momentum
    data['high_3d'] = data['high'].rolling(window=3, min_periods=1).max()
    data['low_3d'] = data['low'].rolling(window=3, min_periods=1).min()
    data['short_scale_momentum'] = ((data['close'] - data['close'].shift(3)) / (data['high_3d'] - data['low_3d'] + 0.001)) * (data['volume'] / (data['volume'].shift(3) + 0.001))
    
    data['high_6d'] = data['high'].rolling(window=6, min_periods=1).max()
    data['low_6d'] = data['low'].rolling(window=6, min_periods=1).min()
    data['medium_scale_momentum'] = ((data['close'] - data['close'].shift(6)) / (data['high_6d'] - data['low_6d'] + 0.001)) * (data['volume'] / (data['volume'].shift(6) + 0.001))
    
    data['scale_alignment_score'] = data['ultra_short_momentum'] * data['short_scale_momentum'] * data['medium_scale_momentum'] * np.sign(data['short_scale_momentum'] - data['medium_scale_momentum'])
    
    # Volatility Regime Patterns
    data['upper_volatility_regime'] = ((data['high'] - data['open']) / (data['open'] - data['low'] + 0.001)) * (data['volume'] / (data['volume'].shift(1) + 0.001))
    data['lower_volatility_regime'] = ((data['close'] - data['low']) / (data['high'] - data['close'] + 0.001)) * (data['volume'] / (data['volume'].shift(1) + 0.001))
    data['volatility_regime_ratio'] = data['upper_volatility_regime'] / (data['lower_volatility_regime'] + 0.001)
    data['regime_volatility_signal'] = data['volatility_regime_ratio'] * np.sign(data['close'] - data['close'].shift(1)) * ((data['high'] - data['low']) / (data['high'].shift(2) - data['low'].shift(2) + 0.001))
    
    # Momentum Persistence Detection
    def count_persistence(series, window=3):
        persistence = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            count = 0
            for j in range(i-window+1, i):
                if j > 0 and j-1 > 0:
                    if np.sign(series.iloc[j] - series.iloc[j-1]) == np.sign(series.iloc[j-1] - series.iloc[j-2]):
                        count += 1
            persistence.iloc[i] = count
        return persistence
    
    data['volume_momentum_persistence'] = count_persistence(data['volume'])
    data['price_momentum_persistence'] = count_persistence(data['close'])
    volatility_range = data['high'] - data['low']
    data['volatility_momentum_persistence'] = count_persistence(volatility_range)
    data['persistence_convergence'] = data['volume_momentum_persistence'] * data['price_momentum_persistence'] * data['volatility_momentum_persistence'] * np.sign(data['price_momentum_persistence'] - data['volatility_momentum_persistence'])
    
    # Adaptive Momentum Construction
    data['transition_core'] = data['regime_transition_score'] * data['acceleration_convergence']
    data['gap_core'] = data['gap_momentum_ratio'] * data['regime_volatility_signal']
    data['scale_core'] = data['scale_alignment_score'] * data['persistence_convergence']
    
    data['regime_momentum'] = data['regime_transition_score'] * data['scale_alignment_score']
    data['persistence_momentum'] = data['persistence_convergence'] * data['acceleration_convergence']
    
    data['base_adaptive_alpha'] = data['transition_core'] * data['gap_core'] * data['scale_core']
    data['enhanced_adaptive_alpha'] = data['base_adaptive_alpha'] * data['regime_momentum'] * (1 + np.abs(data['persistence_momentum']))
    data['final_alpha'] = data['enhanced_adaptive_alpha'] * np.sign(data['regime_transition_score']) * np.sign(data['persistence_convergence'])
    
    return data['final_alpha']
