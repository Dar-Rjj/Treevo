import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price-based features
    data['close_5d_ago'] = data['close'].shift(5)
    data['close_20d_ago'] = data['close'].shift(20)
    data['close_3d_ago'] = data['close'].shift(3)
    data['close_8d_ago'] = data['close'].shift(8)
    
    # True Range calculation
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    
    # Multi-Timeframe Momentum Assessment
    data['momentum_5d'] = data['close'] / data['close_5d_ago'] - 1
    data['momentum_20d'] = data['close'] / data['close_20d_ago'] - 1
    data['momentum_acceleration'] = (data['momentum_5d'] - data['momentum_20d']) / (data['momentum_5d'] + 1e-8)
    
    # Price Efficiency Analysis
    data['daily_price_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Rolling calculations for efficiency persistence and mean reversion
    data['efficiency_persistence'] = data['daily_price_efficiency'].rolling(window=5, min_periods=1).sum()
    
    data['close_5d_avg'] = data['close'].rolling(window=5, min_periods=1).mean()
    data['close_5d_std'] = data['close'].rolling(window=5, min_periods=1).std()
    data['multi_scale_mean_reversion'] = (data['close'] - data['close_5d_avg']) / (data['close_5d_std'] + 1e-8)
    
    # Efficiency-Momentum Integration
    data['momentum_quality_score'] = data['momentum_acceleration'] * data['efficiency_persistence']
    data['reversion_adjustment'] = data['momentum_quality_score'] * data['multi_scale_mean_reversion']
    
    # Volatility context
    data['avg_true_range_5d'] = data['true_range'].rolling(window=5, min_periods=1).mean()
    data['volatility_context'] = data['reversion_adjustment'] / (data['avg_true_range_5d'] / data['close'] + 1e-8)
    
    # Microstructure Regime Detection - Spread Efficiency Analysis
    data['spread_efficiency_3d'] = (data['close'] / data['close_3d_ago'] - 1) / ((data['high'] - data['low']) / data['close'] + 1e-8)
    
    # Price impact efficiency momentum
    data['volume_8d_avg'] = data['volume'].rolling(window=8, min_periods=1).mean()
    data['price_impact_efficiency_8d'] = (data['close'] / data['close_8d_ago'] - 1) / (data['volume'] / (data['volume_8d_avg'] + 1e-8) + 1e-8)
    
    # Regime Classification
    data['true_range_5d_avg'] = data['true_range'].rolling(window=5, min_periods=1).mean()
    data['true_range_20d_avg'] = data['true_range'].rolling(window=20, min_periods=1).mean()
    data['spread_ratio'] = data['true_range_5d_avg'] / (data['true_range_20d_avg'] + 1e-8)
    
    # Regime classification
    data['regime'] = 'normal'
    data.loc[data['spread_ratio'] > 1.2, 'regime'] = 'high_stress'
    data.loc[data['spread_ratio'] < 0.8, 'regime'] = 'low_stress'
    
    # Regime Strength Assessment
    data['spread_price_alignment'] = np.sign(data['spread_efficiency_3d']) * np.sign(data['price_impact_efficiency_8d'])
    
    data['volume_4d_avg'] = data['volume'].rolling(window=4, min_periods=1).mean()
    data['liquidity_resilience_score'] = (data['volume'] / (data['volume_4d_avg'] + 1e-8)) / ((data['high'] - data['low']) / data['close'] + 1e-8)
    
    # Regime persistence strength
    regime_map = {'high_stress': 1, 'low_stress': 2, 'normal': 3}
    data['regime_numeric'] = data['regime'].map(regime_map)
    data['regime_persistence_strength'] = data['regime_numeric'].rolling(window=5, min_periods=1).apply(
        lambda x: (x == x[-1]).sum() / len(x) if len(x) > 0 else 0
    )
    
    # Regime-Adaptive Signal Synthesis
    # Microstructure Confirmation Component
    data['microstructure_confirmation'] = 0.0
    high_stress_mask = data['regime'] == 'high_stress'
    low_stress_mask = data['regime'] == 'low_stress'
    normal_mask = data['regime'] == 'normal'
    
    data.loc[high_stress_mask, 'microstructure_confirmation'] = (
        data.loc[high_stress_mask, 'spread_price_alignment'] * 
        data.loc[high_stress_mask, 'regime_persistence_strength']
    )
    data.loc[low_stress_mask, 'microstructure_confirmation'] = (
        -data.loc[low_stress_mask, 'liquidity_resilience_score'] * 
        data.loc[low_stress_mask, 'regime_persistence_strength']
    )
    data.loc[normal_mask, 'microstructure_confirmation'] = (
        data.loc[normal_mask, 'regime_persistence_strength'] * 
        data.loc[normal_mask, 'spread_price_alignment']
    )
    
    # Regime-Specific Weighting
    data['momentum_efficiency_weight'] = 0.5
    data.loc[high_stress_mask, 'momentum_efficiency_weight'] = 0.3
    data.loc[low_stress_mask, 'momentum_efficiency_weight'] = 0.7
    
    data['microstructure_weight'] = 1 - data['momentum_efficiency_weight']
    
    # Base Signal
    data['base_signal'] = (
        data['momentum_efficiency_weight'] * data['volatility_context'] +
        data['microstructure_weight'] * data['microstructure_confirmation']
    )
    
    # Confirmation Adjustment
    data['confirmation_adjustment'] = data['base_signal'] * data['microstructure_confirmation']
    
    # Volatility Scaling
    data['volatility_scaling'] = data['confirmation_adjustment'] / (data['avg_true_range_5d'] / data['close'] + 1e-8)
    
    # Composite Alpha Generation
    data['core_factor'] = data['volatility_scaling'] * data['momentum_quality_score']
    data['efficiency_validation'] = data['core_factor'] * data['efficiency_persistence']
    data['final_alpha'] = data['efficiency_validation'] * data['regime_persistence_strength']
    
    # Return the final alpha factor
    return data['final_alpha']
