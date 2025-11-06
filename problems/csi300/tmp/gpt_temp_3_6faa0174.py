import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Timeframe Volatility Structure
    data['micro_vol'] = (data['high'] - data['low']) / data['close'].shift(1)
    
    # Rolling windows for meso and macro volatility
    data['high_4d'] = data['high'].rolling(window=5, min_periods=5).max()
    data['low_4d'] = data['low'].rolling(window=5, min_periods=5).min()
    data['meso_vol'] = (data['high_4d'] - data['low_4d']) / data['close'].shift(5)
    
    data['high_20d'] = data['high'].rolling(window=20, min_periods=20).max()
    data['low_20d'] = data['low'].rolling(window=20, min_periods=20).min()
    data['macro_vol'] = (data['high_20d'] - data['low_20d']) / data['close'].shift(20)
    
    # Volume Fractal Analysis
    data['volume_ratio'] = data['volume'] / data['volume'].shift(1)
    data['volume_range_coherence'] = (data['volume'] * (data['high'] - data['low'])) / \
                                   (data['volume'].shift(1) * (data['high'].shift(1) - data['low'].shift(1)))
    
    # Multi-scale Volume Momentum
    data['volume_momentum_short'] = (data['volume'] / data['volume'].shift(3)) ** (1/3)
    data['volume_momentum_long'] = (data['volume'].shift(3) / data['volume'].shift(6)) ** (1/3)
    data['multi_scale_volume_momentum'] = data['volume_momentum_short'] - data['volume_momentum_long']
    
    # Volume Persistence Score
    def volume_persistence(series):
        if len(series) < 3:
            return np.nan
        signs = np.sign(series.diff())
        persistence_count = 0
        for i in range(2, len(signs)):
            if signs.iloc[i] == signs.iloc[i-1] and not np.isnan(signs.iloc[i]) and not np.isnan(signs.iloc[i-1]):
                persistence_count += 1
        return persistence_count / 3
    
    data['volume_persistence'] = data['volume'].rolling(window=3, min_periods=3).apply(
        volume_persistence, raw=False
    )
    
    # Price-Volume Fractal Alignment
    data['micro_alignment'] = np.sign(data['close'] / data['close'].shift(1) - 1) * \
                            np.sign(data['volume'] / data['volume'].shift(1) - 1)
    data['meso_alignment'] = np.sign(data['close'] / data['close'].shift(5) - 1) * \
                           np.sign(data['volume'] / data['volume'].shift(5) - 1)
    data['macro_alignment'] = np.sign(data['close'] / data['close'].shift(20) - 1) * \
                            np.sign(data['volume'] / data['volume'].shift(20) - 1)
    data['fractal_consistency'] = data['micro_alignment'] + data['meso_alignment'] + data['macro_alignment']
    
    # Microstructure Pressure Gradient Analysis
    data['opening_pressure'] = (data['open'] - data['low']) / (data['high'] - data['low'])
    data['closing_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['pressure_shift'] = data['closing_pressure'] - data['opening_pressure']
    data['pressure_momentum'] = data['pressure_shift'] - (data['closing_pressure'].shift(1) - data['opening_pressure'].shift(1))
    
    # Volume-Weighted Pressure
    data['vol_weighted_opening'] = data['opening_pressure'] * data['volume']
    data['vol_weighted_closing'] = data['closing_pressure'] * data['volume']
    data['weighted_pressure_shift'] = data['vol_weighted_closing'] - data['vol_weighted_opening']
    data['weighted_pressure_momentum'] = data['weighted_pressure_shift'] - \
                                       (data['vol_weighted_closing'].shift(1) - data['vol_weighted_opening'].shift(1))
    
    # Price-Level Memory and Reaction Dynamics
    memory_levels = ['close_1', 'close_2', 'close_3', 'close_5', 'close_10', 'close_15']
    for i, lag in enumerate([1, 2, 3, 5, 10, 15]):
        data[f'close_{lag}'] = data['close'].shift(lag)
    
    def count_above(row, levels):
        return sum(1 for level in levels if row['high'] > level) / len(levels)
    
    def count_below(row, levels):
        return sum(1 for level in levels if row['low'] < level) / len(levels)
    
    data['memory_resistance'] = data.apply(lambda x: count_above(x, [x[level] for level in memory_levels]), axis=1)
    data['memory_support'] = data.apply(lambda x: count_below(x, [x[level] for level in memory_levels]), axis=1)
    
    # Price-Level Reaction Intensity
    data['resistance_break_attempt'] = (data['high'] - data[[f'close_{lag}' for lag in [1,2,3,5,10,15]]].max(axis=1)) / \
                                     (data['high'] - data['low'])
    data['support_hold_strength'] = (data[[f'close_{lag}' for lag in [1,2,3,5,10,15]]].min(axis=1) - data['low']) / \
                                  (data['high'] - data['low'])
    data['level_reaction_asymmetry'] = data['resistance_break_attempt'] - data['support_hold_strength']
    data['reaction_momentum'] = data['level_reaction_asymmetry'] - data['level_reaction_asymmetry'].shift(1)
    
    # Volume-Confirmed Level Reactions
    data['resistance_volume_intensity'] = data['volume'] * data['resistance_break_attempt']
    data['support_volume_intensity'] = data['volume'] * data['support_hold_strength']
    data['level_volume_asymmetry'] = data['resistance_volume_intensity'] - data['support_volume_intensity']
    
    # Level Reaction Persistence
    def level_persistence(series):
        if len(series) < 3:
            return np.nan
        signs = np.sign(series)
        persistence_count = 0
        for i in range(2, len(signs)):
            if signs.iloc[i] == signs.iloc[i-1] and not np.isnan(signs.iloc[i]) and not np.isnan(signs.iloc[i-1]):
                persistence_count += 1
        return persistence_count / 3
    
    data['level_reaction_persistence'] = data['level_reaction_asymmetry'].rolling(window=3, min_periods=3).apply(
        level_persistence, raw=False
    )
    
    # Temporal Pattern Coherence
    data['micro_return'] = data['close'] / data['close'].shift(1) - 1
    data['meso_return'] = data['close'] / data['close'].shift(5) - 1
    data['macro_return'] = data['close'] / data['close'].shift(20) - 1
    data['return_scale_coherence'] = data['micro_return'] * data['meso_return'] * data['macro_return']
    
    # Multi-scale Alignment
    data['multi_scale_alignment'] = data['micro_alignment'] + data['meso_alignment'] + data['macro_alignment']
    
    # Pattern Persistence Dynamics
    def return_persistence(series):
        if len(series) < 3:
            return np.nan
        signs = np.sign(series)
        persistence_count = 0
        for i in range(2, len(signs)):
            if signs.iloc[i] == signs.iloc[i-1] and not np.isnan(signs.iloc[i]) and not np.isnan(signs.iloc[i-1]):
                persistence_count += 1
        return persistence_count / 3
    
    data['return_direction_persistence'] = data['micro_return'].rolling(window=3, min_periods=3).apply(
        return_persistence, raw=False
    )
    
    def pressure_persistence(series):
        if len(series) < 3:
            return np.nan
        signs = np.sign(series)
        persistence_count = 0
        for i in range(2, len(signs)):
            if signs.iloc[i] == signs.iloc[i-1] and not np.isnan(signs.iloc[i]) and not np.isnan(signs.iloc[i-1]):
                persistence_count += 1
        return persistence_count / 3
    
    data['pressure_direction_persistence'] = data['pressure_shift'].rolling(window=3, min_periods=3).apply(
        pressure_persistence, raw=False
    )
    
    data['overall_pattern_consistency'] = (data['return_direction_persistence'] + 
                                         data['volume_persistence'] + 
                                         data['pressure_direction_persistence']) / 3
    
    # Core Fractal Components
    data['microstructure_pressure_momentum'] = data['pressure_momentum'] * data['volume']
    data['price_level_reaction_momentum'] = data['reaction_momentum'] * data['level_volume_asymmetry']
    data['multi_scale_alignment_momentum'] = data['multi_scale_alignment'] * data['return_scale_coherence']
    data['fractal_volume_coherence'] = data['multi_scale_volume_momentum'] * data['fractal_consistency']
    
    # Regime-Specific Enhancement
    data['vol_ratio_micro_meso'] = data['micro_vol'] / data['meso_vol']
    data['vol_ratio_meso_macro'] = data['meso_vol'] / data['macro_vol']
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=5).mean()
    
    # Coherence-Validated Signals
    data['pressure_alignment_coherence'] = data['microstructure_pressure_momentum'] * data['multi_scale_alignment']
    data['level_reaction_coherence'] = data['price_level_reaction_momentum'] * data['fractal_consistency']
    data['volume_return_coherence'] = data['fractal_volume_coherence'] * data['multi_scale_alignment']
    data['pattern_persistence_coherence'] = data['multi_scale_alignment_momentum'] * data['overall_pattern_consistency']
    
    # Dynamic Threshold Adaptation
    data['micro_vol_20d_avg'] = data['micro_vol'].rolling(window=20, min_periods=20).mean()
    
    # Final Alpha Synthesis with regime weighting
    primary_factor = data['pressure_alignment_coherence'] * data['volume_persistence']
    secondary_factor = data['level_reaction_coherence'] * data['level_reaction_persistence']
    tertiary_factor = data['volume_return_coherence'] * data['return_direction_persistence']
    quaternary_factor = data['pattern_persistence_coherence'] * data['pressure_direction_persistence']
    
    # Regime-based weighting
    high_vol_regime = (data['vol_ratio_micro_meso'] > 1.2).astype(float)
    low_vol_regime = (data['vol_ratio_meso_macro'] < 0.8).astype(float)
    volume_surge = (data['volume'] > 1.5 * data['volume_5d_avg']).astype(float)
    volume_drought = (data['volume'] < 0.7 * data['volume_5d_avg']).astype(float)
    
    # Apply regime adjustments
    primary_factor = primary_factor * (1 + 0.4 * high_vol_regime)
    secondary_factor = secondary_factor * (1 - 0.2 * low_vol_regime)
    tertiary_factor = tertiary_factor * (1 + 0.3 * volume_surge - 0.3 * volume_drought)
    
    # Adaptive weights
    vol_adaptive_weight = 1 / (data['micro_vol'] + 1e-8)
    vol_weight = data['volume'] / (data['volume_5d_avg'] + 1e-8)
    persistence_weight = data['overall_pattern_consistency']
    
    # Final composite alpha
    composite_alpha = (
        primary_factor * vol_adaptive_weight +
        secondary_factor * vol_weight +
        tertiary_factor * persistence_weight +
        quaternary_factor * data['fractal_consistency']
    ) / 4
    
    return composite_alpha
