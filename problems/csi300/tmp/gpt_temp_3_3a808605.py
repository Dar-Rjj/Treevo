import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Aware Multi-Timeframe Alpha Factor
    Combines momentum acceleration across timeframes with volume confirmation and volatility adjustment,
    dynamically adapting to market regimes.
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Multi-Timeframe Momentum Acceleration
    # Short-term Acceleration (1-3 days)
    data['ret_1d'] = data['close'] / data['close'].shift(1) - 1
    data['ret_3d'] = data['close'] / data['close'].shift(3) - 1
    data['short_accel'] = (data['ret_1d'] - data['ret_3d']/3) / (abs(data['ret_3d']/3) + 0.001)
    data['short_weighted'] = data['short_accel'] * (1 + abs(data['ret_1d']))
    
    # Medium-term Acceleration (5-10 days)
    data['ret_5d'] = data['close'] / data['close'].shift(5) - 1
    data['ret_10d'] = data['close'] / data['close'].shift(10) - 1
    data['medium_accel'] = (data['ret_5d'] - data['ret_10d']/2) / (abs(data['ret_10d']/2) + 0.001)
    
    # Multi-timeframe Alignment
    data['direction_consistency'] = np.sign(data['short_accel']) * np.sign(data['medium_accel'])
    short_norm = data['short_accel'] / (data['short_accel'].abs().rolling(20, min_periods=1).mean() + 0.001)
    medium_norm = data['medium_accel'] / (data['medium_accel'].abs().rolling(20, min_periods=1).mean() + 0.001)
    data['alignment_strength'] = short_norm * medium_norm
    data['composite_accel'] = data['direction_consistency'] * (data['short_weighted'] + data['medium_accel']) / 2
    
    # Volume Intensity Confirmation
    # Volume Momentum Analysis
    data['vol_mom_3d'] = data['volume'] / data['volume'].shift(3) - 1
    data['vol_mom_10d'] = data['volume'] / data['volume'].shift(10) - 1
    data['vol_accel'] = (data['vol_mom_3d'] - data['vol_mom_10d']/3.33) / (abs(data['vol_mom_10d']/3.33) + 0.001)
    data['vol_regime'] = data['volume'] / data['volume'].rolling(20, min_periods=1).median()
    
    # Volume-Price Synchronization
    data['vol_weighted_mom'] = data['composite_accel'] * (1 + data['vol_accel'])
    data['vol_amplified'] = np.where(data['vol_regime'] > 1, 
                                   data['vol_weighted_mom'] * (1 + data['vol_accel']),
                                   data['vol_weighted_mom'] * data['vol_regime'])
    
    # Volatility-Adjusted Returns
    # Adaptive Volatility Estimation
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['vol_proxy'] = data['true_range'] / data['close'].shift(1)
    data['vol_5d_ma'] = data['vol_proxy'].rolling(5, min_periods=1).mean()
    data['vol_regime_ratio'] = data['vol_proxy'] / data['vol_proxy'].rolling(20, min_periods=1).median()
    
    # Risk-Scaled Momentum
    data['vol_adjusted_mom'] = data['vol_amplified'] / (data['vol_5d_ma'] + 0.001)
    data['vol_regime_adjusted'] = np.where(data['vol_regime_ratio'] > 1,
                                         data['vol_adjusted_mom'] / (1 + data['vol_regime_ratio']),
                                         data['vol_adjusted_mom'] * (1 + (1 - data['vol_regime_ratio'])))
    
    # Regime-Adaptive Blending
    # Market State Detection
    data['vol_regime_conf'] = abs(data['vol_regime_ratio'] - 1)
    data['volume_regime_conf'] = abs(data['vol_regime'] - 1)
    data['trend_magnitude'] = (data['close'] / data['close'].shift(20) - 1).abs()
    data['trend_consistency'] = data['ret_1d'].rolling(20, min_periods=1).apply(
        lambda x: len([i for i in range(1, len(x)) if np.sign(x.iloc[i]) == np.sign(x.iloc[i-1])]) / max(len(x)-1, 1)
    )
    data['trend_regime_conf'] = (1 - data['trend_consistency']) * data['trend_magnitude']
    data['overall_regime_conf'] = (data['vol_regime_conf'] + data['volume_regime_conf'] + data['trend_regime_conf']) / 3
    
    # Dynamic Factor Composition
    data['regime_adaptive_weight'] = 1 / (1 + data['overall_regime_conf'])
    data['regime_adjusted_short'] = data['short_weighted'] * data['regime_adaptive_weight']
    data['regime_adjusted_medium'] = data['medium_accel'] * (1 - data['regime_adaptive_weight'])
    
    # Final factor: blend short-term and medium-term based on regime stability
    data['final_factor'] = data['vol_regime_adjusted'] * data['regime_adaptive_weight'] + \
                          data['regime_adjusted_short'] * data['regime_adaptive_weight'] + \
                          data['regime_adjusted_medium'] * (1 - data['regime_adaptive_weight'])
    
    return data['final_factor']
