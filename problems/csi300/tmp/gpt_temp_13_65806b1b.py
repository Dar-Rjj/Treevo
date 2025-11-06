import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility Regime Classification
    data['range_vol'] = (data['high'] - data['low']) / data['close']
    data['gap_vol'] = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Regime conditions
    high_vol_cond = (data['range_vol'] > 0.03) & (data['range_vol'] > data['range_vol'].shift(3))
    low_vol_cond = (data['range_vol'] < 0.015) & (data['range_vol'] < data['range_vol'].shift(3))
    
    # Multi-Scale Fractal Momentum calculations
    def calculate_fractal_ratio(window_size):
        range_sum = data['high'].rolling(window=window_size).apply(lambda x: np.sum(np.abs(x - data.loc[x.index, 'low'])), raw=False)
        close_diff_sum = abs(data['close'] - data['close'].shift(1)).rolling(window=window_size).sum()
        return np.log(range_sum + 1e-8) / np.log(close_diff_sum + 1e-8)
    
    # Short-term (3-5 day)
    data['fractal_3d'] = calculate_fractal_ratio(3)
    data['fractal_5d'] = calculate_fractal_ratio(5)
    data['fractal_momentum_short'] = data['fractal_5d'] - data['fractal_3d']
    
    # Medium-term (8-13 day)
    data['fractal_8d'] = calculate_fractal_ratio(8)
    data['fractal_13d'] = calculate_fractal_ratio(13)
    data['fractal_momentum_medium'] = data['fractal_13d'] - data['fractal_8d']
    
    # Long-term (15-21 day)
    data['fractal_15d'] = calculate_fractal_ratio(15)
    data['fractal_21d'] = calculate_fractal_ratio(21)
    data['fractal_momentum_long'] = data['fractal_21d'] - data['fractal_15d']
    
    # Volume Momentum Components
    data['volume_momentum_short'] = data['volume'] / data['volume'].shift(3)
    data['volume_momentum_medium'] = data['volume'] / data['volume'].shift(8)
    data['volume_momentum_long'] = data['volume'] / data['volume'].shift(15)
    
    # Regime-Adaptive Processing
    # High Volatility Processing
    data['fractal_decay_short'] = data['fractal_momentum_short'] / (1 + data['range_vol'])
    data['volume_confirmation_short'] = data['volume_momentum_short'] * abs(data['fractal_momentum_short'])
    
    data['fractal_decay_medium'] = data['fractal_momentum_medium'] / (1 + data['range_vol'])
    data['volume_confirmation_medium'] = data['volume_momentum_medium'] * abs(data['fractal_momentum_medium'])
    
    data['fractal_decay_long'] = data['fractal_momentum_long'] / (1 + data['range_vol'])
    data['volume_confirmation_long'] = data['volume_momentum_long'] * abs(data['fractal_momentum_long'])
    
    # Low Volatility Processing
    data['fractal_persistence_short'] = (data['fractal_momentum_short'] * 
                                        np.sign(data['fractal_momentum_short'].shift(1)) * 
                                        np.sign(data['fractal_momentum_short'].shift(2)))
    data['volume_acceleration_short'] = (data['volume_momentum_short'] + data['volume_momentum_medium']) / 2
    
    data['fractal_persistence_medium'] = (data['fractal_momentum_medium'] * 
                                         np.sign(data['fractal_momentum_medium'].shift(1)) * 
                                         np.sign(data['fractal_momentum_medium'].shift(2)))
    data['volume_acceleration_medium'] = (data['volume_momentum_medium'] + data['volume_momentum_long']) / 2
    
    data['fractal_persistence_long'] = (data['fractal_momentum_long'] * 
                                       np.sign(data['fractal_momentum_long'].shift(1)) * 
                                       np.sign(data['fractal_momentum_long'].shift(2)))
    data['volume_acceleration_long'] = (data['volume_momentum_long'] + data['volume_momentum_short']) / 2
    
    # Transition Processing
    data['balanced_signal_short'] = (data['fractal_decay_short'] * data['volume_confirmation_short'] + 
                                    data['fractal_persistence_short'] * data['volume_acceleration_short']) / 2
    data['direction_alignment_short'] = (np.sign(data['fractal_momentum_short']) * 
                                        np.sign(data['volume_momentum_short']) * 
                                        np.sign(data['amount'] / data['amount'].shift(3) - 1))
    
    data['balanced_signal_medium'] = (data['fractal_decay_medium'] * data['volume_confirmation_medium'] + 
                                     data['fractal_persistence_medium'] * data['volume_acceleration_medium']) / 2
    data['direction_alignment_medium'] = (np.sign(data['fractal_momentum_medium']) * 
                                         np.sign(data['volume_momentum_medium']) * 
                                         np.sign(data['amount'] / data['amount'].shift(3) - 1))
    
    data['balanced_signal_long'] = (data['fractal_decay_long'] * data['volume_confirmation_long'] + 
                                   data['fractal_persistence_long'] * data['volume_acceleration_long']) / 2
    data['direction_alignment_long'] = (np.sign(data['fractal_momentum_long']) * 
                                       np.sign(data['volume_momentum_long']) * 
                                       np.sign(data['amount'] / data['amount'].shift(3) - 1))
    
    # Multi-Timeframe Validation
    # Cross-Scale Alignment
    fractal_signs = pd.DataFrame({
        'short': np.sign(data['fractal_momentum_short']),
        'medium': np.sign(data['fractal_momentum_medium']),
        'long': np.sign(data['fractal_momentum_long'])
    })
    data['scale_alignment'] = fractal_signs.apply(lambda x: sum(x == x.iloc[0]), axis=1)
    
    volume_fractal_alignment = pd.DataFrame({
        'short': np.sign(data['volume_momentum_short']) == np.sign(data['fractal_momentum_short']),
        'medium': np.sign(data['volume_momentum_medium']) == np.sign(data['fractal_momentum_medium']),
        'long': np.sign(data['volume_momentum_long']) == np.sign(data['fractal_momentum_long'])
    })
    data['volume_fractal_alignment'] = volume_fractal_alignment.sum(axis=1)
    
    data['cross_scale_alignment'] = (data['scale_alignment'] + data['volume_fractal_alignment']) / 6
    
    # Regime-Adaptive Base Signal
    base_signals = []
    for timeframe in ['short', 'medium', 'long']:
        fractal_momentum = data[f'fractal_momentum_{timeframe}']
        volume_momentum = data[f'volume_momentum_{timeframe}']
        
        high_vol_signal = (data[f'fractal_decay_{timeframe}'] * data[f'volume_confirmation_{timeframe}'])
        low_vol_signal = (data[f'fractal_persistence_{timeframe}'] * data[f'volume_acceleration_{timeframe}'])
        transition_signal = (data[f'balanced_signal_{timeframe}'] * data[f'direction_alignment_{timeframe}'])
        
        base_signal = np.where(high_vol_cond, high_vol_signal,
                              np.where(low_vol_cond, low_vol_signal, transition_signal))
        
        base_signals.append(base_signal)
    
    # Multi-Scale Integration
    weights = [0.4, 0.35, 0.25]
    weighted_base = sum(w * signal for w, signal in zip(weights, base_signals))
    
    # Temporal Consistency
    base_df = pd.DataFrame({'base': weighted_base}, index=data.index)
    data['temporal_consistency'] = base_df['base'].rolling(window=3).apply(
        lambda x: sum(np.sign(x)), raw=False
    )
    
    # Final Alpha Calculation
    data['quality_adjusted_signal'] = weighted_base * (1 + data['cross_scale_alignment'] * 0.15)
    
    # Opening Efficiency
    data['opening_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Final Alpha with volatility enhancement
    final_alpha = data['quality_adjusted_signal'] * (1 + data['opening_efficiency'])
    
    return final_alpha
