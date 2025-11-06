import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Timeframe Fractal Momentum
    data['fractal_3d'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    data['fractal_5d'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['fractal_10d'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    
    # Fractal Acceleration Components
    data['ultra_short_acc'] = (data['fractal_3d'] - data['fractal_5d']) / 2
    data['short_term_acc'] = (data['fractal_5d'] - data['fractal_10d']) / 5
    
    # Momentum Sustainability
    momentum_sign = np.sign(data['fractal_5d'])
    momentum_sustainability = pd.Series(index=data.index, dtype=float)
    for i in range(5, len(data)):
        window = momentum_sign.iloc[i-4:i+1]
        if len(window) == 5:
            if (window == window.iloc[0]).all():
                momentum_sustainability.iloc[i] = 5
            else:
                count = 1
                for j in range(1, 5):
                    if window.iloc[j] == window.iloc[j-1]:
                        count += 1
                    else:
                        break
                momentum_sustainability.iloc[i] = count
    
    # Volume-Enhanced Acceleration
    data['volume_momentum_acc'] = ((data['volume'] / data['volume'].shift(1) - 1) - 
                                  (data['volume'].shift(1) / data['volume'].shift(2) - 1))
    data['acceleration_strength'] = data['fractal_5d'] * (data['ultra_short_acc'] + data['short_term_acc']) / 2
    data['volume_aligned_acc'] = data['acceleration_strength'] * data['volume_momentum_acc']
    
    # Microstructure Pressure Dynamics
    data['buying_pressure'] = ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)) * data['volume']
    data['selling_pressure'] = ((data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)) * data['volume']
    data['net_pressure'] = data['buying_pressure'] - data['selling_pressure']
    
    # Execution Efficiency System
    data['gap_efficiency'] = np.abs((data['close'] - data['open']) / 
                                   (np.abs((data['open'] / data['close'].shift(1) - 1)) * data['close'].shift(1) + 1e-8))
    data['flow_quality'] = ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)) * \
                          (1 - np.abs(data['amount'] / data['amount'].shift(1) - 1))
    data['depth_pressure'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Pressure-Efficiency Integration
    data['pressure_direction'] = np.sign(data['net_pressure'])
    data['efficiency_multiplier'] = data['gap_efficiency'] * data['flow_quality']
    data['microstructure_signal'] = data['net_pressure'] * data['efficiency_multiplier'] * data['depth_pressure']
    
    # Regime-Adaptive Divergence Framework
    data['price_volume_alignment'] = (np.sign(data['fractal_3d']) * 
                                     np.sign(data['volume'] - data['volume'].shift(3)) * 
                                     np.abs(data['fractal_3d'] * (data['volume'] - data['volume'].shift(3))))
    data['divergence_strength'] = np.abs(data['price_volume_alignment'])
    data['divergence_direction'] = np.sign(data['price_volume_alignment'])
    
    # Regime Classification
    data['regime'] = 'normal'
    data.loc[data['divergence_strength'] > 0.7, 'regime'] = 'strong'
    data.loc[data['divergence_strength'] < 0.3, 'regime'] = 'weak'
    
    # Regime Stability Components
    regime_changes = (data['regime'] != data['regime'].shift(1)).astype(int)
    data['regime_stability'] = 1 - (regime_changes.rolling(window=5, min_periods=1).sum() / 5)
    
    volume_median_10d = data['volume'].rolling(window=10, min_periods=1).median()
    data['volume_persistence'] = (data['volume'] > volume_median_10d).rolling(window=5, min_periods=1).sum() / 5
    
    close_change_sign = np.sign(data['close'] - data['close'].shift(1))
    flow_consistency = pd.Series(index=data.index, dtype=float)
    for i in range(5, len(data)):
        window = close_change_sign.iloc[i-4:i+1]
        if len(window) == 5:
            if (window == window.iloc[0]).all():
                flow_consistency.iloc[i] = 5
            else:
                count = 1
                for j in range(1, 5):
                    if window.iloc[j] == window.iloc[j-1]:
                        count += 1
                    else:
                        break
                flow_consistency.iloc[i] = count
    
    # Volatility-Enhanced Breakout Dynamics
    data['upper_breakout'] = (data['close'] - data['high'].rolling(window=5, min_periods=1).max()) * \
                            np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['lower_breakout'] = (data['close'] - data['low'].rolling(window=5, min_periods=1).min()) * \
                            (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['range_breakout_score'] = data['upper_breakout'] + data['lower_breakout']
    
    # Volatility Cluster Detection
    data['daily_range'] = (data['high'] - data['low']) / ((data['high'] + data['low']) / 2 + 1e-8)
    range_avg_20d = data['daily_range'].rolling(window=20, min_periods=1).mean()
    
    volatility_cluster = pd.Series(0, index=data.index)
    cluster_intensity = pd.Series(0.0, index=data.index)
    
    for i in range(20, len(data)):
        if all(data['daily_range'].iloc[i-2:i+1] > range_avg_20d.iloc[i]):
            volatility_cluster.iloc[i] = 1
            cluster_period = data['daily_range'].iloc[i-2:i+1]
            volume_avg_20d = data['volume'].rolling(window=20, min_periods=1).mean().iloc[i]
            cluster_intensity.iloc[i] = cluster_period.sum() * data['volume'].iloc[i] / volume_avg_20d
    
    # Breakout Acceleration
    data['price_impact_ratio'] = ((data['close'] - data['close'].shift(1)) / data['close'].shift(1) * data['volume']) / \
                                (np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8) + 1e-8)
    data['volatility_adjusted_breakout'] = data['range_breakout_score'] / (data['daily_range'] + 1e-8)
    data['cluster_enhanced_breakout'] = data['volatility_adjusted_breakout'] * cluster_intensity
    
    # Adaptive Factor Construction
    data['core_acceleration'] = data['volume_aligned_acc'] * momentum_sustainability
    data['quality_adjustment'] = flow_consistency * (1 - np.abs(data['price_volume_alignment']))
    data['base_factor'] = data['core_acceleration'] * data['quality_adjustment'] * data['divergence_direction']
    
    # Regime-Adaptive Enhancement
    enhancement = pd.Series(1.0, index=data.index)
    enhancement[data['regime'] == 'strong'] = 1 + data['regime_stability']
    enhancement[data['regime'] == 'weak'] = 1 - data['regime_stability']
    data['regime_enhanced_factor'] = data['base_factor'] * enhancement
    
    # Microstructure Integration
    data['pressure_confirmation'] = data['microstructure_signal'] * data['pressure_direction']
    data['breakout_alignment'] = data['cluster_enhanced_breakout'] * data['price_impact_ratio']
    data['enhanced_factor'] = data['regime_enhanced_factor'] * data['pressure_confirmation'] * data['breakout_alignment']
    
    # Volume Surge Override System
    volume_median_10d = data['volume'].rolling(window=10, min_periods=1).median()
    data['volume_surge_condition'] = data['volume'] / volume_median_10d > 2.0
    data['volume_leadership'] = (data['volume'] / volume_median_10d) * \
                               ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8))
    data['surge_persistence'] = data['volume_persistence'] * data['volume_leadership']
    
    # Final Adaptive Factor
    data['core_signal'] = data['enhanced_factor'] * data['surge_persistence']
    data['volume_confirmation'] = data['volume'] / data['volume'].rolling(window=10, min_periods=1).mean()
    
    # Regime-dependent scaling
    regime_scaling = pd.Series(1.0, index=data.index)
    regime_scaling[data['regime'] == 'strong'] = 1.2
    regime_scaling[data['regime'] == 'weak'] = 0.8
    
    data['regime_adaptive_alpha'] = data['core_signal'] * data['volume_confirmation'] * regime_scaling
    
    return data['regime_adaptive_alpha']
