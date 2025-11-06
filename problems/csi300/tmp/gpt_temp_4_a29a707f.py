import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Core Gap Components
    df['volatility_scaled_gap'] = (df['open'] / df['close'].shift(1) - 1) / df['high'].rolling(5).apply(lambda x: (x - x.shift(1)).mean(), raw=False)
    df['gap_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['gap_persistence'] = (df['open'] / df['close'].shift(1) - 1) * (df['close'] / df['open'] - 1)
    df['raw_gap_divergence'] = df['gap_persistence'] - (df['volume'] / df['volume'].shift(1) - 1)
    
    # Gap Divergence Dynamics
    df['gap_divergence_momentum'] = df['raw_gap_divergence'] / df['raw_gap_divergence'].rolling(2).mean() - 1
    df['gap_divergence_acceleration'] = df['raw_gap_divergence'] - df['raw_gap_divergence'].rolling(3).mean()
    
    def gap_divergence_persistence(series):
        if len(series) < 3:
            return np.nan
        current_sign = np.sign(series.iloc[-1])
        prev_signs = np.sign(series.iloc[-3:-1])
        return np.sum(prev_signs == current_sign)
    
    df['gap_divergence_persistence'] = df['raw_gap_divergence'].rolling(3).apply(gap_divergence_persistence, raw=False)
    
    # Volume Pressure Analysis
    df['volume_efficiency'] = (df['volume'] / df['volume'].rolling(5).mean()) * (1 / ((df['high'] - df['low']) / df['volume']))
    df['net_pressure'] = (df['close'] - df['low']) - (df['high'] - df['close'])
    df['volume_intensity'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_pressure_divergence'] = df['volume_efficiency'] * df['net_pressure']
    
    # Asymmetry Detection
    df['positive_gap_divergence'] = np.where(df['raw_gap_divergence'] > 0, df['raw_gap_divergence'], 0)
    df['negative_gap_divergence'] = np.where(df['raw_gap_divergence'] < 0, df['raw_gap_divergence'], 0)
    df['gap_asymmetry_ratio'] = df['positive_gap_divergence'] / (np.abs(df['negative_gap_divergence']) + 0.001)
    
    def volume_asymmetry_ratio(volume_series, close_series):
        if len(volume_series) < 10:
            return np.nan
        up_volume = np.sum(volume_series.iloc[-10:] * (close_series.iloc[-10:] > close_series.iloc[-10:].shift(1)))
        down_volume = np.sum(volume_series.iloc[-10:] * (close_series.iloc[-10:] < close_series.iloc[-10:].shift(1)))
        return up_volume / (down_volume + 0.001)
    
    df['volume_asymmetry_ratio'] = df['volume'].rolling(10).apply(lambda x: volume_asymmetry_ratio(x, df['close'].loc[x.index]), raw=False)
    df['pressure_asymmetry'] = df['net_pressure'] / df['net_pressure'].abs().rolling(5).mean()
    df['combined_asymmetry'] = df['gap_asymmetry_ratio'] * df['volume_asymmetry_ratio']
    
    # Multi-Scale Fractal Analysis
    df['gap_momentum_2d'] = df['volatility_scaled_gap'] / df['volatility_scaled_gap'].rolling(2).mean() - 1
    df['gap_acceleration_3d'] = df['volatility_scaled_gap'] - df['volatility_scaled_gap'].rolling(3).mean()
    df['short_term_asymmetry'] = df['gap_asymmetry_ratio'] / df['gap_asymmetry_ratio'].rolling(3).mean()
    
    df['gap_trend_5d'] = df['volatility_scaled_gap'].rolling(5).mean()
    df['gap_stability_10d'] = df['volatility_scaled_gap'].rolling(10).std()
    df['medium_term_asymmetry_shift'] = df['gap_asymmetry_ratio'] - df['gap_asymmetry_ratio'].rolling(10).mean()
    
    # Range & Breakout Enhancement
    df['net_breakout'] = (df['close'] - df['high'].rolling(5).max()) - (df['low'].rolling(5).min() - df['close'])
    df['gap_compression_ratio'] = np.abs(df['open'] / df['close'].shift(1) - 1) / (np.abs(df['open'].shift(8) / df['close'].shift(9) - 1) + 0.001)
    df['range_efficiency'] = (df['high'] - df['low']) / (df['high'] - df['low']).rolling(10).mean()
    
    df['gap_breakout_alignment'] = df['raw_gap_divergence'] * df['net_breakout']
    df['compression_divergence_interaction'] = df['raw_gap_divergence'] * df['gap_compression_ratio']
    df['range_adjusted_divergence'] = df['raw_gap_divergence'] / df['range_efficiency']
    
    # Amount-Based Confirmation
    df['trade_size_indicator'] = df['amount'] / df['volume']
    df['size_momentum'] = df['trade_size_indicator'] / df['trade_size_indicator'].rolling(5).mean() - 1
    df['size_pressure_alignment'] = df['net_pressure'] * df['size_momentum']
    
    df['size_gap_confirmation'] = df['raw_gap_divergence'] * df['size_momentum']
    df['amount_intensity'] = df['amount'] / df['amount'].rolling(20).mean()
    df['amount_weighted_divergence'] = df['raw_gap_divergence'] * df['amount_intensity']
    
    # Volatility-Regime Classification
    gap_volatility = df['volatility_scaled_gap'].rolling(20).std()
    df['high_gap_regime'] = np.abs(df['volatility_scaled_gap']) > gap_volatility
    df['low_gap_regime'] = np.abs(df['volatility_scaled_gap']) < (0.5 * gap_volatility)
    df['normal_gap_regime'] = ~(df['high_gap_regime'] | df['low_gap_regime'])
    
    df['extreme_gap_reversal'] = df['volatility_scaled_gap'] / gap_volatility
    df['volatility_amplified_divergence'] = df['range_adjusted_divergence'] * (df['high'] - df['low']) / df['close']
    
    df['gap_accumulation'] = df['volatility_scaled_gap'].rolling(5).sum()
    df['stability_enhanced_divergence'] = df['raw_gap_divergence'] / (df['gap_stability_10d'] + 0.001)
    
    df['regime_change_detection'] = df['high_gap_regime'] != df['high_gap_regime'].shift(1)
    df['transition_momentum'] = df['raw_gap_divergence'] * df['regime_change_detection']
    
    # Price-Level Adaptive Framework
    df['price_level'] = df['close']
    df['price_level_momentum'] = df['close'] / df['close'].rolling(10).mean() - 1
    df['level_adjusted_gap'] = df['volatility_scaled_gap'] / (df['price_level_momentum'] + 0.001)
    
    df['daily_price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 0.001)
    df['position_momentum'] = df['daily_price_position'] - df['daily_price_position'].rolling(5).mean()
    df['position_gap_interaction'] = df['volatility_scaled_gap'] * df['position_momentum']
    
    # Composite Alpha Generation
    df['gap_divergence_core'] = df['raw_gap_divergence'] * df['gap_divergence_momentum'] * df['gap_divergence_acceleration']
    df['volume_pressure_enhancement'] = df['gap_divergence_core'] * df['volume_pressure_divergence']
    df['asymmetry_amplification'] = df['volume_pressure_enhancement'] * df['combined_asymmetry']
    
    df['fractal_timing'] = df['asymmetry_amplification'] * df['short_term_asymmetry'] * df['medium_term_asymmetry_shift']
    df['breakout_confirmation'] = df['fractal_timing'] * df['gap_breakout_alignment'] * df['compression_divergence_interaction']
    df['amount_validation'] = df['breakout_confirmation'] * df['size_gap_confirmation'] * df['amount_weighted_divergence']
    
    # Regime-Adaptive Weighting
    regime_multiplier = np.where(df['high_gap_regime'], df['volatility_amplified_divergence'],
                                np.where(df['low_gap_regime'], df['stability_enhanced_divergence'],
                                        df['transition_momentum']))
    
    df['regime_adjusted_momentum'] = df['amount_validation'] * regime_multiplier * df['gap_compression_ratio'] * df['volume_intensity']
    
    # Final Alpha Factor
    df['gap_efficiency_adjusted'] = df['regime_adjusted_momentum'] * df['gap_efficiency']
    df['price_level_adapted'] = df['gap_efficiency_adjusted'] * df['level_adjusted_gap']
    df['final_alpha'] = df['price_level_adapted'] / df['range_efficiency']
    
    return df['final_alpha']
