import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Overnight Gap Momentum
    data['overnight_return'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    def calculate_gap_fill_probability(row):
        if row['overnight_return'] > 0:
            return 1 - abs((row['open'] - data['low'].shift(1).loc[row.name]) / 
                          (data['high'].shift(1).loc[row.name] - data['low'].shift(1).loc[row.name]))
        else:
            return abs((row['open'] - data['high'].shift(1).loc[row.name]) / 
                      (data['high'].shift(1).loc[row.name] - data['low'].shift(1).loc[row.name]))
    
    data['gap_fill_probability'] = data.apply(calculate_gap_fill_probability, axis=1)
    data['gap_momentum_score'] = data['overnight_return'] * (1 - data['gap_fill_probability'])
    
    # Volume Acceleration Regimes
    data['volume_acceleration'] = (data['volume'] / data['volume'].shift(1)) - (data['volume'].shift(1) / data['volume'].shift(2))
    
    def calculate_volume_regime(row):
        accel = row['volume_acceleration']
        accel_prev = data['volume_acceleration'].shift(1).loc[row.name]
        if pd.isna(accel) or pd.isna(accel_prev):
            return 0
        return np.sign(accel) * (abs(accel) / (abs(accel) + abs(accel_prev)))
    
    data['volume_regime'] = data.apply(calculate_volume_regime, axis=1)
    
    # Regime Persistence
    def calculate_regime_persistence(row, window=5):
        current_idx = data.index.get_loc(row.name)
        if current_idx < window:
            return 0
        regime_window = data['volume_regime'].iloc[current_idx-window+1:current_idx+1]
        if len(regime_window) < 2:
            return 0
        persistence_count = sum(np.sign(regime_window.iloc[i]) == np.sign(regime_window.iloc[i-1]) 
                              for i in range(1, len(regime_window)))
        return persistence_count / window
    
    data['regime_persistence'] = data.apply(calculate_regime_persistence, axis=1)
    
    # Price-Volume Divergence
    data['price_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, 1e-10)
    data['volume_momentum'] = (data['volume'] - data['volume'].shift(1)) / data['volume'].shift(1).replace(0, 1e-10)
    data['divergence_score'] = data['price_momentum'] * data['volume_momentum']
    
    # Divergence Strength
    data['price_momentum_std'] = data['price_momentum'].rolling(window=10, min_periods=1).std()
    data['volume_momentum_std'] = data['volume_momentum'].rolling(window=10, min_periods=1).std()
    data['divergence_strength'] = abs(data['divergence_score']) / (data['price_momentum_std'] * data['volume_momentum_std']).replace(0, 1e-10)
    
    # Microstructure Noise Ratio
    data['true_range_efficiency'] = (data['high'] - data['low']) / (
        abs(data['open'] - data['close'].shift(1)) + 
        abs(data['close'] - data['open']) + 
        abs(data['high'] - data['low'])
    ).replace(0, 1e-10)
    
    data['volume_ma_10'] = data['volume'].rolling(window=10, min_periods=1).mean()
    data['volume_std_10'] = data['volume'].rolling(window=10, min_periods=1).std()
    data['volume_noise'] = (data['volume'] - data['volume_ma_10']) / data['volume_std_10'].replace(0, 1e-10)
    
    data['noise_ratio'] = data['true_range_efficiency'] / (1 + abs(data['volume_noise']))
    data['signal_quality'] = 1 - data['noise_ratio']
    
    # Multi-Timeframe Volume Pressure
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_ma_20'] = data['volume'].rolling(window=20, min_periods=1).mean()
    data['short_term_pressure'] = data['volume'] / data['volume_ma_5'].replace(0, 1e-10)
    data['medium_term_pressure'] = data['volume'] / data['volume_ma_20'].replace(0, 1e-10)
    data['volume_pressure_ratio'] = data['short_term_pressure'] / data['medium_term_pressure'].replace(0, 1e-10)
    data['pressure_momentum'] = data['volume_pressure_ratio'] - data['volume_pressure_ratio'].shift(3)
    
    # Regime Transition Signals
    data['gap_regime_interaction'] = data['gap_momentum_score'] * data['volume_regime']
    data['divergence_regime_signal'] = data['divergence_score'] * data['regime_persistence']
    data['noise_adaptive_momentum'] = data['price_momentum'] * data['signal_quality']
    data['pressure_transition'] = data['pressure_momentum'] * data['volume_regime']
    
    # Cross-Timeframe Alpha Generation
    data['gap_driven_alpha'] = data['gap_regime_interaction'] * (1 + data['divergence_strength'])
    data['noise_weighted_alpha'] = data['noise_adaptive_momentum'] * data['signal_quality']
    data['pressure_based_alpha'] = data['pressure_transition'] * data['volume_pressure_ratio']
    data['regime_adaptive_alpha'] = data['divergence_regime_signal'] * (1 + data['regime_persistence'])
    
    # Dynamic Alpha Synthesis
    denominator = (abs(data['gap_momentum_score']) + abs(data['signal_quality']) + abs(data['volume_pressure_ratio'])).replace(0, 1e-10)
    data['gap_weight'] = abs(data['gap_momentum_score']) / denominator
    data['noise_weight'] = abs(data['signal_quality']) / denominator
    data['pressure_weight'] = abs(data['volume_pressure_ratio']) / denominator
    
    remaining_weight = 1 - (data['gap_weight'] + data['noise_weight'] + data['pressure_weight'])
    
    # Final Alpha
    data['final_alpha'] = (
        data['gap_driven_alpha'] * data['gap_weight'] +
        data['noise_weighted_alpha'] * data['noise_weight'] +
        data['pressure_based_alpha'] * data['pressure_weight'] +
        data['regime_adaptive_alpha'] * remaining_weight
    )
    
    return data['final_alpha']
