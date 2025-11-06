import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Calculate daily returns
    data['returns'] = data['close'].pct_change()
    
    # Volatility Regime & Asymmetry Identification
    data['vol_5d'] = data['returns'].rolling(window=5).std()
    data['vol_20d'] = data['returns'].rolling(window=20).std()
    data['vol_ratio'] = data['vol_5d'] / data['vol_20d']
    
    # Price pressure asymmetry
    data['upside_pressure'] = (data['high'] - data['open']) / data['open']
    data['downside_pressure'] = (data['open'] - data['low']) / data['open']
    data['asymmetry_ratio'] = data['upside_pressure'] / (data['downside_pressure'] + 1e-8)
    
    # Regime-Adaptive Asymmetric Acceleration
    # High volatility regime
    high_vol_mask = data['vol_ratio'] > 1
    data['price_acceleration_high_vol'] = np.nan
    data['price_acceleration_low_vol'] = np.nan
    
    # High vol acceleration
    data['range_3d'] = data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min()
    data['price_acceleration_high_vol'] = ((data['close'] - data['close'].shift(3)) / 
                                         (data['range_3d'] + 1e-8)) * data['asymmetry_ratio']
    
    # Low vol acceleration
    data['ret_5d'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['ret_20d'] = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)
    data['price_acceleration_low_vol'] = (data['ret_5d'] - data['ret_20d']) * data['asymmetry_ratio']
    
    # Combine regime-adaptive acceleration
    data['regime_acceleration'] = np.where(high_vol_mask, 
                                         data['price_acceleration_high_vol'], 
                                         data['price_acceleration_low_vol'])
    
    # Volume Concentration Quality Dynamics
    data['avg_trade_size'] = data['amount'] / (data['volume'] + 1e-8)
    data['trade_size_concentration'] = data['avg_trade_size'].rolling(window=10).std()
    data['volume_persistence'] = (data['volume'].rolling(window=3).sum() / 
                                data['volume'].rolling(window=8).sum())
    data['volume_quality_raw'] = data['avg_trade_size'] * data['volume_persistence']
    data['volume_quality'] = data['volume_quality_raw'] - data['volume_quality_raw'].rolling(window=5).mean()
    
    # Multi-timeframe Acceleration Measurement
    data['pressure_asymmetry_3d'] = data['asymmetry_ratio'].rolling(window=3).mean()
    data['pressure_asymmetry_8d'] = data['asymmetry_ratio'].rolling(window=8).mean()
    data['pressure_acceleration'] = data['pressure_asymmetry_3d'] - data['pressure_asymmetry_8d']
    
    data['volume_quality_3d'] = data['volume_quality'].rolling(window=3).mean()
    data['volume_quality_8d'] = data['volume_quality'].rolling(window=8).mean()
    data['volume_quality_acceleration'] = data['volume_quality_3d'] - data['volume_quality_8d']
    
    # Price momentum calculation
    data['price_momentum'] = data['close'].pct_change(periods=3)
    data['volume_momentum'] = data['volume'].pct_change(periods=3)
    data['momentum_divergence'] = (np.sign(data['price_momentum'] * data['volume_momentum']) * 
                                 np.abs(data['price_momentum']))
    
    # Range-Constrained Pattern Enhancement
    data['daily_amplitude'] = (data['high'] - data['low']) / ((data['high'] + data['low']) / 2 + 1e-8)
    data['range_persistence_momentum'] = data['daily_amplitude'].diff(periods=5)
    
    # Multi-timeframe alignment
    data['momentum_aligned'] = ((data['price_momentum'] > 0) & (data['volume_momentum'] > 0) | 
                              (data['price_momentum'] < 0) & (data['volume_momentum'] < 0))
    data['alignment_count'] = data['momentum_aligned'].rolling(window=5).sum()
    
    # Asymmetry-Concentration Alignment Patterns
    data['upside_vol_corr'] = data['upside_pressure'].rolling(window=10).corr(data['volume'])
    data['downside_vol_corr'] = data['downside_pressure'].rolling(window=10).corr(data['volume'])
    
    # Extreme pressure concentration
    data['pressure_extreme'] = ((data['asymmetry_ratio'] > data['asymmetry_ratio'].rolling(window=20).quantile(0.8)) | 
                              (data['asymmetry_ratio'] < data['asymmetry_ratio'].rolling(window=20).quantile(0.2)))
    
    # Alignment confirmation
    data['alignment_confirmation'] = ((data['upside_vol_corr'] > 0.3) & (data['pressure_extreme']) & 
                                    (data['alignment_count'] >= 3)).astype(float)
    
    # Misalignment penalty
    data['misalignment_penalty'] = ((data['upside_vol_corr'] < -0.3) | 
                                  (data['downside_vol_corr'] < -0.3)).astype(float) * 0.5
    
    # Convergence-Divergence Pattern Detection
    data['range_change_3d'] = data['daily_amplitude'].diff(periods=3)
    
    # Zero crossings calculation
    def count_zero_crossings(series):
        signs = np.sign(series)
        return ((signs != signs.shift(1)) & (signs.shift(1) != 0)).sum()
    
    data['oscillation_frequency'] = data['range_change_3d'].rolling(window=15).apply(
        count_zero_crossings, raw=False)
    
    # Final Alpha Generation
    # Base signal
    base_signal = data['regime_acceleration']
    
    # Volume quality multiplier
    volume_multiplier = 1 + data['volume_quality'] / (np.abs(data['volume_quality']).rolling(window=20).mean() + 1e-8)
    
    # Multi-timeframe acceleration adjustment
    acceleration_adjustment = data['pressure_acceleration'] + data['volume_quality_acceleration']
    
    # Range constraint
    range_constraint = 1 / (data['daily_amplitude'] + 1e-8)
    
    # Alignment confirmation factor
    alignment_factor = 1 + data['alignment_confirmation'] * 0.2 - data['misalignment_penalty']
    
    # Pattern enhancement
    convergence_strength = (data['alignment_count'] / 5) * (1 - data['oscillation_frequency'] / 15)
    pattern_enhancement = 1 + convergence_strength * 0.3
    
    # Combine all components
    alpha = (base_signal * volume_multiplier + acceleration_adjustment) * range_constraint
    alpha = alpha * alignment_factor * pattern_enhancement
    
    return alpha
