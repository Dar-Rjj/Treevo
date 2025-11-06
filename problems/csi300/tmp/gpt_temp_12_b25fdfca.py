import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Momentum Structure
    # Hierarchical Momentum Measures
    data['ultra_short_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['short_term_momentum'] = data['close'] / data['close'].shift(5) - 1
    data['medium_term_momentum'] = data['close'] / data['close'].shift(20) - 1
    
    # Fractal Momentum Properties
    data['micro_scale_momentum'] = ((data['high'] - data['low']) / data['close'].shift(1)) * data['ultra_short_momentum']
    
    # Rolling calculations for meso and macro scales
    data['high_9d'] = data['high'].rolling(window=10, min_periods=10).max()
    data['low_9d'] = data['low'].rolling(window=10, min_periods=10).min()
    data['meso_scale_momentum'] = ((data['high_9d'] - data['low_9d']) / data['close'].shift(10)) * data['short_term_momentum']
    
    data['high_19d'] = data['high'].rolling(window=20, min_periods=20).max()
    data['low_19d'] = data['low'].rolling(window=20, min_periods=20).min()
    data['macro_scale_momentum'] = ((data['high_19d'] - data['low_19d']) / data['close'].shift(20)) * data['medium_term_momentum']
    
    # Momentum Quality Assessment
    # Momentum Consistency
    momentum_consistency = []
    for i in range(len(data)):
        if i < 5:
            momentum_consistency.append(np.nan)
        else:
            count = 0
            for j in range(i-4, i+1):
                if np.sign(data['ultra_short_momentum'].iloc[j]) == np.sign(data['short_term_momentum'].iloc[j]):
                    count += 1
            momentum_consistency.append(count / 5)
    data['momentum_consistency'] = momentum_consistency
    
    # Fractal Momentum Alignment
    data['fractal_momentum_alignment'] = np.sign(data['micro_scale_momentum']) * np.sign(data['meso_scale_momentum']) * np.sign(data['macro_scale_momentum'])
    
    # Momentum Persistence
    momentum_persistence = []
    for i in range(len(data)):
        if i < 5:
            momentum_persistence.append(np.nan)
        else:
            count = 0
            for j in range(i-4, i+1):
                if j > 0 and np.sign(data['ultra_short_momentum'].iloc[j]) == np.sign(data['ultra_short_momentum'].iloc[j-1]):
                    count += 1
            momentum_persistence.append(count / 5)
    data['momentum_persistence'] = momentum_persistence
    
    # Order Flow Momentum Integration
    # Volume-Momentum Interaction
    data['volume_weighted_momentum'] = data['ultra_short_momentum'] * (data['volume'] / data['volume'].shift(1))
    data['flow_momentum'] = (data['close'] - (data['high'] + data['low'])/2) * data['volume'] * data['ultra_short_momentum']
    data['opening_momentum_pressure'] = (data['open'] - data['close'].shift(1)) * data['volume'] * data['ultra_short_momentum']
    
    # Money Flow Momentum
    data['directional_flow_momentum'] = data['amount'] * data['ultra_short_momentum']
    data['flow_intensity_momentum'] = data['directional_flow_momentum'] / (data['high'] - data['low'] + 1e-8)
    
    # Flow Persistence
    flow_persistence = []
    for i in range(len(data)):
        if i < 3:
            flow_persistence.append(np.nan)
        else:
            count = 0
            for j in range(i-2, i+1):
                if j > 0 and np.sign(data['directional_flow_momentum'].iloc[j]) == np.sign(data['directional_flow_momentum'].iloc[j-1]):
                    count += 1
            flow_persistence.append(count / 3)
    data['flow_persistence'] = flow_persistence
    
    # Transaction Momentum Structure
    data['avg_trade_size'] = data['amount'] / (data['volume'] + 1e-8)
    data['trade_size_momentum'] = data['avg_trade_size'] / data['avg_trade_size'].shift(1) - 1
    data['volume_concentration_momentum'] = (data['volume'] / (data['avg_trade_size'] + 1e-8)) * data['ultra_short_momentum']
    data['transaction_clustering_momentum'] = (data['volume'] / data['volume'].shift(1)) * data['trade_size_momentum'] * data['ultra_short_momentum']
    
    # Efficiency-Momentum Coupling
    # Price Discovery Momentum
    data['information_efficiency_momentum'] = ((data['close'] - data['open'])**2 / ((data['high'] - data['low'] + 1e-8) * data['volume'])) * data['ultra_short_momentum']
    data['price_impact_momentum'] = ((data['high'] - data['low']) / (data['volume'] + 1e-8)) * data['ultra_short_momentum']
    data['market_depth_momentum'] = (data['volume'] / (data['high'] - data['low'] + 1e-8)) * data['ultra_short_momentum']
    
    # Efficiency Dynamics Integration
    data['efficiency_momentum_persistence'] = data['information_efficiency_momentum'] / (data['information_efficiency_momentum'].shift(1) + 1e-8) - 1
    
    # Depth Momentum Persistence
    depth_persistence = []
    for i in range(len(data)):
        if i < 5:
            depth_persistence.append(np.nan)
        else:
            count = 0
            for j in range(i-4, i+1):
                if j > 0 and data['market_depth_momentum'].iloc[j] > data['market_depth_momentum'].iloc[j-1]:
                    count += 1
            depth_persistence.append(count / 5)
    data['depth_momentum_persistence'] = depth_persistence
    
    data['impact_reversal_momentum'] = np.sign(data['price_impact_momentum'] / (data['price_impact_momentum'].shift(1) + 1e-8) - 1) * np.sign(data['ultra_short_momentum'])
    
    # Fractal Efficiency Momentum
    data['multi_scale_efficiency_momentum'] = data['information_efficiency_momentum'] * (data['micro_scale_momentum'] / (data['meso_scale_momentum'] + 1e-8))
    data['range_adaptive_efficiency'] = data['information_efficiency_momentum'] / (data['micro_scale_momentum'] + 1e-8)
    data['depth_weighted_flow_momentum'] = data['market_depth_momentum'] * data['flow_intensity_momentum']
    
    # Momentum Regime Detection
    # Volatility-Momentum Regimes
    data['micro_scale_ma_5d'] = data['micro_scale_momentum'].rolling(window=5, min_periods=5).mean()
    data['high_volatility_momentum'] = data['micro_scale_momentum'] > (1.5 * data['micro_scale_ma_5d'])
    data['low_volatility_momentum'] = data['micro_scale_momentum'] < (0.7 * data['micro_scale_ma_5d'])
    data['normal_volatility_momentum'] = ~(data['high_volatility_momentum'] | data['low_volatility_momentum'])
    
    # Trend-Momentum Regimes
    data['trend_strength_ratio'] = np.abs(data['short_term_momentum']) / (data['meso_scale_momentum'] + 0.001)
    data['strong_trend_momentum'] = data['trend_strength_ratio'] > 2.0
    data['weak_trend_momentum'] = data['trend_strength_ratio'] < 0.5
    data['normal_trend_momentum'] = ~(data['strong_trend_momentum'] | data['weak_trend_momentum'])
    
    # Flow-Momentum Regimes
    data['flow_intensity_ma_5d'] = data['flow_intensity_momentum'].rolling(window=5, min_periods=5).mean()
    data['high_flow_momentum'] = data['flow_intensity_momentum'] > (1.5 * data['flow_intensity_ma_5d'])
    data['low_flow_momentum'] = data['flow_intensity_momentum'] < (0.7 * data['flow_intensity_ma_5d'])
    data['normal_flow_momentum'] = ~(data['high_flow_momentum'] | data['low_flow_momentum'])
    
    # Composite Momentum Construction
    # Core Momentum Synthesis
    data['fractal_momentum_alpha'] = data['micro_scale_momentum'] * data['meso_scale_momentum'] * data['fractal_momentum_alignment']
    data['flow_momentum_alpha'] = data['volume_weighted_momentum'] * data['flow_intensity_momentum'] * data['flow_persistence']
    data['efficiency_momentum_alpha'] = data['multi_scale_efficiency_momentum'] * data['efficiency_momentum_persistence']
    data['transaction_momentum_alpha'] = data['volume_concentration_momentum'] * data['transaction_clustering_momentum']
    
    # Regime-Adaptive Momentum
    data['volatility_regime_momentum'] = data['fractal_momentum_alpha'] * (1 + 0.5 * data['high_volatility_momentum'] - 0.3 * data['low_volatility_momentum'])
    data['trend_regime_momentum'] = data['flow_momentum_alpha'] * (1 + 0.4 * data['strong_trend_momentum'] - 0.2 * data['weak_trend_momentum'])
    data['flow_regime_momentum'] = data['efficiency_momentum_alpha'] * (1 + 0.3 * data['high_flow_momentum'] - 0.2 * data['low_flow_momentum'])
    data['multi_regime_momentum'] = data['transaction_momentum_alpha'] * (data['high_volatility_momentum'] | data['strong_trend_momentum'] | data['high_flow_momentum'])
    
    # Momentum Validation Enhancement
    data['consistency_boost'] = np.where(data['momentum_consistency'] > 0.6, 1.25, 1.0)
    data['alignment_boost'] = np.where(data['fractal_momentum_alignment'] > 0, 1.3, 1.0)
    data['persistence_boost'] = np.where(data['momentum_persistence'] > 0.6, 1.2, 1.0)
    data['flow_confirmation_boost'] = np.where(data['flow_persistence'] > 0.6, 1.15, 1.0)
    
    # Asymmetric Momentum Response
    # Directional Momentum Factors
    data['up_move_condition'] = data['close'] > data['close'].shift(2)
    data['down_move_condition'] = data['close'] < data['close'].shift(2)
    
    data['up_move_momentum'] = np.where(data['up_move_condition'], data['fractal_momentum_alpha'], 0)
    data['down_move_momentum'] = np.where(data['down_move_condition'], data['fractal_momentum_alpha'], 0)
    data['flow_asymmetry'] = np.where(data['up_move_condition'], data['flow_momentum_alpha'], -data['flow_momentum_alpha'])
    data['efficiency_asymmetry'] = np.where(data['up_move_condition'], data['efficiency_momentum_alpha'], -data['efficiency_momentum_alpha'])
    
    # Asymmetric Momentum Integration
    data['directional_momentum_ratio'] = data['up_move_momentum'] / (data['down_move_momentum'] + 1e-8)
    data['flow_directional_momentum'] = data['flow_asymmetry'] * data['directional_momentum_ratio']
    data['efficiency_directional_momentum'] = data['efficiency_asymmetry'] * data['directional_momentum_ratio']
    
    # Final Fractal Momentum Alpha
    # Regime-Based Selection
    data['high_volatility_alpha'] = data['volatility_regime_momentum'] * 1.4 + data['flow_regime_momentum'] * 1.2
    data['strong_trend_alpha'] = data['trend_regime_momentum'] * 1.3 + data['multi_regime_momentum'] * 1.1
    data['normal_regime_alpha'] = (data['volatility_regime_momentum'] + data['trend_regime_momentum'] + 
                                  data['flow_regime_momentum'] + data['multi_regime_momentum']) / 4
    
    # Base Regime Alpha selection
    data['base_regime_alpha'] = np.where(
        data['high_volatility_momentum'], data['high_volatility_alpha'],
        np.where(data['strong_trend_momentum'], data['strong_trend_alpha'], data['normal_regime_alpha'])
    )
    
    # Asymmetric Adjustment
    data['directional_enhancement'] = np.where(
        data['up_move_condition'], data['flow_directional_momentum'],
        np.where(data['down_move_condition'], data['efficiency_directional_momentum'],
                (data['flow_directional_momentum'] + data['efficiency_directional_momentum']) / 2)
    )
    
    # Final Alpha with validation boosts
    data['final_alpha'] = (data['base_regime_alpha'] + data['directional_enhancement']) * \
                         data['consistency_boost'] * data['alignment_boost'] * \
                         data['persistence_boost'] * data['flow_confirmation_boost']
    
    return data['final_alpha']
