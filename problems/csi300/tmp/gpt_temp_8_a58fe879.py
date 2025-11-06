import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility-Regime Adaptive Efficiency Momentum
    # Efficiency Components
    data['abs_price_change_5d'] = data['close'].diff(5).abs()
    data['true_range'] = np.maximum(data['high'] - data['low'], 
                                   np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                             abs(data['low'] - data['close'].shift(1))))
    data['total_true_range_5d'] = data['true_range'].rolling(window=5, min_periods=5).sum()
    data['efficiency_ratio'] = data['abs_price_change_5d'] / data['total_true_range_5d']
    
    # Volatility Regime Classification
    data['returns'] = data['close'].pct_change()
    data['volatility_5d'] = data['returns'].rolling(window=5, min_periods=5).std()
    data['volatility_20d_median'] = data['volatility_5d'].rolling(window=20, min_periods=20).median()
    data['volatility_regime'] = np.where(data['volatility_5d'] > data['volatility_20d_median'] * 1.2, 'high',
                                       np.where(data['volatility_5d'] < data['volatility_20d_median'] * 0.8, 'low', 'transition'))
    
    # Regime-Adaptive Momentum Generation
    data['efficiency_momentum_3d'] = data['efficiency_ratio'].diff(3)
    volatility_ratio = data['volatility_5d'] / data['volatility_20d_median']
    
    def regime_scaling(row):
        if row['volatility_regime'] == 'high':
            return row['efficiency_momentum_3d'] * volatility_ratio
        elif row['volatility_regime'] == 'low':
            return row['efficiency_momentum_3d'] / volatility_ratio
        else:
            return row['efficiency_momentum_3d']
    
    data['volatility_adaptive_momentum'] = data.apply(regime_scaling, axis=1)
    
    # Multi-Scale Volume-Price Harmony
    # Volume Momentum Analysis
    data['volume_momentum_2d'] = data['volume'].pct_change(2)
    data['volume_momentum_6d'] = data['volume'].pct_change(6)
    data['volume_harmony_ratio'] = data['volume_momentum_2d'] / (data['volume_momentum_6d'] + 1e-8)
    
    # Price Reversal Confirmation
    data['daily_reversal_magnitude'] = abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)
    data['reversal_persistence_3d'] = data['daily_reversal_magnitude'].rolling(window=3, min_periods=3).mean()
    
    # Volume-Price Integration
    data['volume_weighted_reversal'] = data['daily_reversal_magnitude'] * data['volume_momentum_2d']
    data['volume_harmony_adjusted'] = data['volume_weighted_reversal'] * data['volume_harmony_ratio']
    data['multi_scale_confirmation'] = data['volume_harmony_adjusted'] * data['reversal_persistence_3d']
    
    # Range Expansion Dynamics with Acceleration
    # Range Analysis Components
    data['daily_range_pct'] = (data['high'] - data['low']) / data['close']
    data['range_expansion_ratio_7d'] = data['daily_range_pct'] / data['daily_range_pct'].rolling(window=7, min_periods=7).mean()
    data['range_acceleration'] = data['range_expansion_ratio_7d'].diff(3)
    
    # Volume Intensity Assessment
    data['volume_intensity'] = data['volume'] / (data['high'] - data['low'] + 1e-8)
    data['volume_intensity_rank'] = data['volume_intensity'].rolling(window=5, min_periods=5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    data['volume_intensity_median_10d'] = data['volume_intensity'].rolling(window=10, min_periods=10).median()
    data['significant_volume_event'] = data['volume_intensity'] > data['volume_intensity_median_10d']
    
    # Dynamic Range-Volume Integration
    data['range_volume_combined'] = data['range_acceleration'] * data['volume_intensity']
    data['range_expansion_multiplier'] = np.where(
        data['significant_volume_event'], 
        data['range_volume_combined'] * 1.5, 
        data['range_volume_combined']
    )
    
    # Regime-Aware Factor Synthesis
    # Core Signal Integration
    data['core_signal'] = (data['volatility_adaptive_momentum'] * 
                          data['multi_scale_confirmation'] * 
                          data['range_expansion_multiplier'] * 
                          data['volume_intensity_rank'])
    
    # Regime-Specific Refinement
    def regime_refinement(row):
        if row['volatility_regime'] == 'high':
            return row['core_signal'] * (1 + abs(row['efficiency_momentum_3d']) + abs(row['range_acceleration']))
        elif row['volatility_regime'] == 'low':
            return row['core_signal'] * (1 + row['reversal_persistence_3d'] + row['volume_harmony_ratio'])
        else:
            return row['core_signal']
    
    data['regime_refined_signal'] = data.apply(regime_refinement, axis=1)
    
    # Final Alpha Generation
    data['volume_trend_3d'] = data['volume'].rolling(window=3, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)
    
    data['final_alpha'] = (data['regime_refined_signal'] * 
                          (1 + data['volume_trend_3d']) * 
                          data['reversal_persistence_3d'])
    
    return data['final_alpha']
