import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Structure
    data['short_term_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['medium_term_momentum'] = data['close'] / data['close'].shift(5) - 1
    data['long_term_momentum'] = data['close'] / data['close'].shift(20) - 1
    
    # Volatility-Volume Coherence Patterns
    data['daily_range'] = data['high'] - data['low']
    data['volatility_ratio'] = data['daily_range'] / data['daily_range'].shift(1)
    data['volatility_momentum'] = data['volatility_ratio'] - data['volatility_ratio'].shift(1)
    
    # Volatility Persistence
    def calc_volatility_persistence(series):
        if len(series) < 4:
            return np.nan
        current_sign = np.sign(series.iloc[-1] - 1)
        prev_signs = [np.sign(series.iloc[i] - 1) for i in range(-3, -1)]
        matches = sum(1 for sign in prev_signs if sign == current_sign)
        return matches / 3
    
    volatility_persistence = data['volatility_ratio'].rolling(window=4, min_periods=4).apply(
        calc_volatility_persistence, raw=False
    )
    data['volatility_persistence'] = volatility_persistence
    
    # Volume Dynamics
    data['volume_momentum'] = data['volume'] / data['volume'].shift(1) - 1
    data['volume_volatility_ratio'] = data['volume'] / data['daily_range']
    data['volume_range_coherence'] = (data['volume'] * data['daily_range']) / (
        data['volume'].shift(1) * data['daily_range'].shift(1)
    )
    
    # Coherence Validation
    data['momentum_volatility_alignment'] = np.sign(data['short_term_momentum']) * np.sign(data['volatility_momentum'])
    data['volume_volatility_divergence'] = np.sign(data['volume_momentum']) * np.sign(data['volatility_momentum'])
    data['range_efficiency_coherence'] = (abs(data['close'] - data['open']) / data['daily_range']) * data['volume_volatility_ratio']
    
    # Price-Level Asymmetry Analysis
    data['intraday_position'] = (data['close'] - data['low']) / data['daily_range']
    data['position_momentum'] = data['intraday_position'] - data['intraday_position'].shift(1)
    data['position_extremes'] = abs(data['intraday_position'] - 0.5) * 2
    
    # Gap Asymmetry
    data['overnight_gap'] = data['open'] / data['close'].shift(1) - 1
    data['gap_filling'] = (data['close'] - data['open']) * np.sign(data['overnight_gap'])
    data['gap_momentum'] = data['overnight_gap'] - data['overnight_gap'].shift(1)
    
    # Breakout Patterns
    data['prev_high_max'] = data['high'].shift(1).rolling(window=2).max()
    data['prev_low_min'] = data['low'].shift(1).rolling(window=2).min()
    data['high_breakout'] = data['high'] / data['prev_high_max'] - 1
    data['low_breakout'] = data['low'] / data['prev_low_min'] - 1
    data['breakout_asymmetry'] = data['high_breakout'] - data['low_breakout']
    
    # Multi-Scale Signal Integration
    data['volatility_adjusted_momentum'] = data['short_term_momentum'] * data['volatility_ratio']
    data['volume_weighted_momentum'] = data['medium_term_momentum'] * data['volume_momentum']
    data['coherence_enhanced_momentum'] = data['long_term_momentum'] * data['range_efficiency_coherence']
    
    data['position_momentum_alignment'] = data['position_momentum'] * data['short_term_momentum']
    data['gap_momentum_interaction'] = data['gap_filling'] * data['medium_term_momentum']
    data['breakout_position_correlation'] = data['breakout_asymmetry'] * data['position_extremes']
    
    # Persistence calculations
    def calc_momentum_persistence(series):
        if len(series) < 4:
            return np.nan
        current_sign = np.sign(series.iloc[-1])
        prev_signs = [np.sign(series.iloc[i]) for i in range(-3, -1)]
        matches = sum(1 for sign in prev_signs if sign == current_sign)
        return matches / 3
    
    def calc_volume_persistence(series):
        if len(series) < 4:
            return np.nan
        current_sign = np.sign(series.iloc[-1])
        prev_signs = [np.sign(series.iloc[i]) for i in range(-3, -1)]
        matches = sum(1 for sign in prev_signs if sign == current_sign)
        return matches / 3
    
    momentum_persistence = data['short_term_momentum'].rolling(window=4, min_periods=4).apply(
        calc_momentum_persistence, raw=False
    )
    volume_persistence = data['volume_momentum'].rolling(window=4, min_periods=4).apply(
        calc_volume_persistence, raw=False
    )
    
    data['momentum_persistence'] = momentum_persistence
    data['volume_persistence'] = volume_persistence
    
    # Composite Alpha Construction
    data['volatility_coherent_momentum'] = data['volatility_adjusted_momentum'] * data['momentum_volatility_alignment']
    data['volume_aligned_momentum'] = data['volume_weighted_momentum'] * data['volume_volatility_divergence']
    data['position_enhanced_momentum'] = data['coherence_enhanced_momentum'] * data['position_momentum_alignment']
    
    data['gap_enhanced_momentum'] = data['volatility_coherent_momentum'] * data['gap_momentum_interaction']
    data['breakout_enhanced_momentum'] = data['volume_aligned_momentum'] * data['breakout_position_correlation']
    data['multi_scale_integration'] = data['position_enhanced_momentum'] * data['gap_enhanced_momentum'] * data['breakout_enhanced_momentum']
    
    # Final Alpha Synthesis
    primary_factor = data['gap_enhanced_momentum'] * data['momentum_persistence']
    secondary_factor = data['breakout_enhanced_momentum'] * data['volatility_persistence']
    tertiary_factor = data['multi_scale_integration'] * data['volume_persistence']
    
    composite_alpha = primary_factor * secondary_factor * tertiary_factor
    
    return composite_alpha
