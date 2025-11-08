import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # Multi-Scale Momentum Fractal Dimensions
    # Short-Scale Momentum (5-day)
    df['momentum_velocity_short'] = df['returns'] - df['returns'].shift(2)
    df['momentum_acceleration_short'] = (df['returns'] - df['returns'].shift(2)) - (df['returns'].shift(2) - df['returns'].shift(4))
    df['momentum_fractality_short'] = np.abs(df['momentum_acceleration_short']) / (np.abs(df['momentum_velocity_short']) + 1e-8)
    
    # Medium-Scale Momentum (15-day)
    df['momentum_velocity_medium'] = df['returns'] - df['returns'].shift(7)
    df['momentum_acceleration_medium'] = (df['returns'] - df['returns'].shift(7)) - (df['returns'].shift(7) - df['returns'].shift(14))
    
    # Calculate consecutive same-sign acceleration periods
    df['acceleration_sign_medium'] = np.sign(df['momentum_acceleration_medium'])
    df['persistence_count'] = 0
    for i in range(1, len(df)):
        if df['acceleration_sign_medium'].iloc[i] == df['acceleration_sign_medium'].iloc[i-1]:
            df.loc[df.index[i], 'persistence_count'] = df['persistence_count'].iloc[i-1] + 1
    
    # Long-Scale Momentum (30-day)
    df['momentum_velocity_long'] = df['returns'] - df['returns'].shift(15)
    df['momentum_acceleration_long'] = (df['returns'] - df['returns'].shift(15)) - (df['returns'].shift(15) - df['returns'].shift(30))
    df['scale_interaction'] = df['momentum_fractality_short'] * df['persistence_count']
    
    # Volume-Price Asymmetry Patterns
    # Volume Directional Asymmetry
    df['up_volume_efficiency'] = np.where(df['close'] > df['open'], 
                                         (df['close'] - df['open']) / (df['volume'] + 1e-8), 0)
    df['down_volume_efficiency'] = np.where(df['close'] < df['open'], 
                                           (df['open'] - df['close']) / (df['volume'] + 1e-8), 0)
    df['volume_asymmetry_ratio'] = df['up_volume_efficiency'] / (df['down_volume_efficiency'] + 1e-8)
    
    # Price Movement Asymmetry
    df['intraday_range_utilization'] = np.abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    df['gap_asymmetry'] = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-8)
    
    # Calculate price path efficiency (sum of absolute returns over 5 days)
    df['abs_returns'] = np.abs(df['returns'])
    df['rolling_abs_sum'] = df['abs_returns'].rolling(window=5, min_periods=1).sum()
    df['price_path_efficiency'] = (df['close'] - df['open']) / (df['rolling_abs_sum'] + 1e-8)
    
    # Microstructure Asymmetry (simplified using volume and amount)
    df['large_trade_impact'] = np.where(df['volume'] > 10000, df['amount'] / (df['volume'] + 1e-8), 0)
    df['small_trade_clustering'] = np.where(df['volume'] < 1000, 1, 0)
    df['trade_size_asymmetry'] = df['large_trade_impact'] / (df['small_trade_clustering'] + 1e-8)
    
    # Fractal-Asymmetry Convergence
    df['short_scale_convergence'] = df['momentum_fractality_short'] * df['volume_asymmetry_ratio']
    df['medium_scale_convergence'] = df['persistence_count'] * df['price_path_efficiency']
    df['multi_scale_alignment'] = df['short_scale_convergence'] * df['medium_scale_convergence'] * df['scale_interaction']
    
    # Microstructure-Enhanced Convergence
    df['trade_size_confirmation'] = df['multi_scale_alignment'] * df['trade_size_asymmetry']
    df['volume_quality_adjustment'] = df['multi_scale_alignment'] * df['volume_asymmetry_ratio']
    df['path_efficiency_weighting'] = df['multi_scale_alignment'] * df['price_path_efficiency']
    
    # Pattern Phase Classification
    df['momentum_phase'] = np.where(
        (np.abs(df['momentum_acceleration_short']) > np.abs(df['momentum_acceleration_medium'])) & 
        (np.abs(df['momentum_acceleration_short']) > np.abs(df['momentum_acceleration_long'])), 1, 0
    )
    df['volume_phase'] = np.where((df['volume_asymmetry_ratio'] > 1.5) | (df['volume_asymmetry_ratio'] < 0.67), 1, 0)
    df['efficiency_phase'] = np.where(df['price_path_efficiency'] > 0.5, 1, 0)
    
    # Phase-Adaptive Signal Weighting
    df['phase_weight'] = np.where(df['momentum_phase'] == 1, 1.3,
                                 np.where(df['volume_phase'] == 1, 1.1, 0.9))
    
    # Generate Multi-Dimensional Composite Factor
    df['core_factor'] = df['multi_scale_alignment'] * df['volume_asymmetry_ratio']
    df['microstructure_integration'] = df['core_factor'] * df['trade_size_asymmetry'] * df['volume_asymmetry_ratio']
    df['phase_enhanced'] = df['microstructure_integration'] * df['phase_weight']
    
    # Apply Dimensional Transformations
    df['exp_scaled'] = np.exp(np.sign(df['phase_enhanced']) * np.log1p(np.abs(df['phase_enhanced'])))
    df['log_compressed'] = np.sign(df['exp_scaled']) * np.log1p(np.abs(df['exp_scaled']))
    
    # Trigonometric modulation for phase transitions
    df['trig_modulation'] = np.sin(df['log_compressed'] * np.pi / 2)
    
    # Raw Multi-Dimensional Score
    df['raw_composite'] = df['trig_modulation'] * df['phase_enhanced']
    
    # Pattern Quality Filters
    df['fractal_quality'] = np.where(
        (df['momentum_fractality_short'] > 0.1) & 
        (df['persistence_count'] > 1), 1, 0
    )
    
    df['asymmetry_quality'] = np.where(
        (df['volume'] > 100000) & 
        (np.abs(df['close'] - df['open']) > 0.001), 1, 0
    )
    
    # Final factor with quality filters
    df['factor'] = df['raw_composite'] * df['fractal_quality'] * df['asymmetry_quality']
    
    return df['factor']
