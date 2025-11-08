import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Weighted Price Asymmetry Analysis
    # Directional Volatility Components
    data['upside_vol'] = np.maximum(0, data['high'] - data['close'].shift(1))
    data['downside_vol'] = np.maximum(0, data['close'].shift(1) - data['low'])
    data['total_vol_range'] = data['high'] - data['low']
    
    # Volatility Asymmetry Ratio
    data['vol_skew_ratio'] = np.log(np.where(data['downside_vol'] > 0, 
                                           data['upside_vol'] / data['downside_vol'], 1.0))
    data['directional_pressure'] = (data['upside_vol'] - data['downside_vol']) / np.where(data['total_vol_range'] > 0, data['total_vol_range'], 1.0)
    data['directional_pressure_ma'] = data['directional_pressure'].rolling(window=3, min_periods=1).mean()
    
    # Intraday Price Rejection
    data['price_rejection'] = np.where(data['high'] != data['low'], 
                                     (data['close'] - data['open']) / (data['high'] - data['low']), 0)
    
    # Volume-Momentum Divergence Analysis
    # Volume Acceleration Patterns
    data['volume_momentum'] = data['volume'] / data['volume'].shift(1) - 1
    data['volume_acceleration'] = (data['volume'] / data['volume'].shift(1)) - (data['volume'].shift(1) / data['volume'].shift(2))
    
    # Volatility-Adjusted Price Momentum
    data['true_range'] = np.maximum(data['high'] - data['low'], 
                                  np.maximum(np.abs(data['high'] - data['close'].shift(1)), 
                                           np.abs(data['low'] - data['close'].shift(1))))
    data['price_return_5d'] = data['close'] / data['close'].shift(5) - 1
    data['vol_adjusted_momentum'] = np.where(data['true_range'] > 0, 
                                           data['price_return_5d'] / data['true_range'], 0)
    
    # Volume-Price Coherence
    data['positive_coherence'] = np.where(data['close'] > data['open'], data['volume'], 0)
    data['negative_coherence'] = np.where(data['close'] < data['open'], data['volume'], 0)
    data['coherence_ratio'] = np.where(data['negative_coherence'] > 0, 
                                     data['positive_coherence'] / data['negative_coherence'], 1.0)
    
    # Market Microstructure Integration
    # Efficiency Regime Indicator
    data['efficiency_indicator'] = np.where(data['high'] != data['low'], 
                                          np.abs(data['close'] - data['open']) / (data['high'] - data['low']), 0)
    
    # Range Utilization Efficiency
    data['range_efficiency'] = np.where(data['high'] != data['low'], 
                                      (data['close'] - data['low']) / (data['high'] - data['low']), 0.5)
    
    # Volatility Regime Analysis
    data['vol_10d'] = data['close'].rolling(window=10, min_periods=1).std()
    data['vol_20d'] = data['close'].rolling(window=20, min_periods=1).std()
    data['vol_ratio'] = np.where(data['vol_20d'] > 0, data['vol_10d'] / data['vol_20d'], 1.0)
    
    # Persistence and Pattern Analysis
    # Asymmetry Persistence
    data['asymmetry_direction'] = np.sign(data['directional_pressure'])
    data['asymmetry_persistence'] = 0
    current_streak = 0
    for i in range(1, len(data)):
        if data['asymmetry_direction'].iloc[i] == data['asymmetry_direction'].iloc[i-1]:
            current_streak += 1
        else:
            current_streak = 0
        data.loc[data.index[i], 'asymmetry_persistence'] = current_streak
    
    # Price Rejection Persistence
    data['rejection_direction'] = np.sign(data['price_rejection'])
    data['rejection_persistence'] = 0
    current_streak = 0
    for i in range(1, len(data)):
        if data['rejection_direction'].iloc[i] == data['rejection_direction'].iloc[i-1]:
            current_streak += 1
        else:
            current_streak = 0
        data.loc[data.index[i], 'rejection_persistence'] = current_streak
    
    data['rejection_strength'] = 1 + 0.05 * data['rejection_persistence']
    
    # Historical Pattern Integration
    data['daily_return'] = data['close'] / data['close'].shift(1) - 1
    data['reversal_pattern'] = 1.0
    for i in range(3, len(data)):
        if i >= 3:
            returns_window = data['daily_return'].iloc[i-2:i+1]
            if len(returns_window) >= 3:
                corr_matrix = np.corrcoef(returns_window.values, 
                                        [returns_window.iloc[0], returns_window.iloc[1], returns_window.iloc[2]])[0,1]
                data.loc[data.index[i], 'reversal_pattern'] = 1 + abs(corr_matrix)
    
    # Volatility-Volume Correlation
    data['vol_volume_corr'] = data['true_range'].rolling(window=5, min_periods=1).corr(data['volume'])
    
    # Final Alpha Factor Construction
    # Core Convergence Component
    data['volume_momentum_divergence'] = data['vol_skew_ratio'] * data['volume_acceleration']
    
    # Microstructure regime weighting
    data['microstructure_weight'] = np.where(data['efficiency_indicator'] > 0.6, 
                                           data['vol_skew_ratio'],  # High efficiency: emphasize price asymmetry
                                           np.where(data['efficiency_indicator'] < 0.4, 
                                                   data['volume_momentum_divergence'],  # Low efficiency: emphasize volume-momentum
                                                   (data['vol_skew_ratio'] + data['volume_momentum_divergence']) / 2))  # Transition: balance both
    
    # Core convergence
    data['core_convergence'] = data['vol_skew_ratio'] * data['volume_momentum_divergence'] * data['microstructure_weight']
    
    # Persistence Quality Adjustment
    data['persistence_quality'] = data['asymmetry_persistence'] * data['efficiency_indicator'] * data['rejection_strength']
    
    # Volatility Regime Integration
    data['vol_regime_multiplier'] = np.where(data['vol_ratio'] > 1.0, 
                                           data['vol_ratio'],  # High volatility regime
                                           0.7)  # Low volatility dampening
    
    # Final Alpha Factor
    data['alpha_factor'] = (data['core_convergence'] * 
                          data['persistence_quality'] * 
                          data['vol_regime_multiplier'] * 
                          data['directional_pressure_ma'])
    
    return data['alpha_factor']
