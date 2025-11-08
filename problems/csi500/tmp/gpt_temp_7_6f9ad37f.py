import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Price Efficiency Dynamics
    data['close_ret'] = data['close'].pct_change()
    data['abs_close_ret'] = abs(data['close_ret'])
    
    # 3-day price efficiency
    data['eff_3d'] = (data['close'] - data['close'].shift(3)) / (
        data['abs_close_ret'].rolling(window=3, min_periods=3).sum()
    )
    
    # 8-day price efficiency
    data['eff_8d'] = (data['close'] - data['close'].shift(8)) / (
        data['abs_close_ret'].rolling(window=8, min_periods=8).sum()
    )
    
    # 5-day absolute price efficiency
    data['eff_5d_abs'] = abs(data['close'] - data['close'].shift(5)) / (
        data['abs_close_ret'].rolling(window=5, min_periods=5).sum()
    )
    
    # Efficiency momentum divergence
    data['eff_mom_div'] = data['eff_3d'] - data['eff_8d']
    
    # Range Elasticity Analysis
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    
    # 3-day range elasticity
    data['range_elasticity_3d'] = (
        data['true_range'] / data['true_range'].rolling(window=3, min_periods=3).mean() - 1
    )
    
    # 10-day range elasticity ratio
    data['range_elasticity_10d'] = (
        data['true_range'].rolling(window=3, min_periods=3).mean() / 
        data['true_range'].rolling(window=10, min_periods=10).mean() - 1
    )
    
    data['elasticity_ratio'] = data['range_elasticity_3d'] / (data['range_elasticity_10d'] + 1e-8)
    
    # Volume Flow Dynamics
    data['volume_accel'] = (
        (data['volume'] - data['volume'].shift(1)) - 
        (data['volume'].shift(1) - data['volume'].shift(2))
    )
    
    data['dollar_volume'] = (data['high'] + data['low'] + data['close']) / 3 * data['volume']
    
    data['volume_mom_div'] = (
        data['volume'] / (data['volume'].shift(1) + 1e-8) - 
        data['volume'].shift(1) / (data['volume'].shift(2) + 1e-8)
    )
    
    data['amount_confidence'] = (
        data['amount'] / data['amount'].rolling(window=5, min_periods=5).mean()
    )
    
    # Price-Flow Alignment Analysis
    data['vwap'] = (data['high'] + data['low'] + data['close']) / 3
    data['price_deviation'] = data['close'] - data['vwap']
    data['intraday_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Bullish/bearish divergence
    data['bullish_div'] = (
        (data['close'] < data['close'].shift(1)) & 
        (data['volume_accel'] > data['volume_accel'].shift(1))
    ).astype(float)
    
    data['bearish_div'] = (
        (data['close'] > data['close'].shift(1)) & 
        (data['volume_accel'] < data['volume_accel'].shift(1))
    ).astype(float)
    
    # Elasticity Regime Classification
    data['atr_5d'] = data['true_range'].rolling(window=5, min_periods=5).mean()
    data['range_vol_5d'] = data['true_range'].rolling(window=5, min_periods=5).std()
    
    # High elasticity regime (top 30%)
    high_elasticity_threshold = data['range_elasticity_3d'].rolling(window=20, min_periods=20).quantile(0.7)
    data['high_elasticity_regime'] = (data['range_elasticity_3d'] > high_elasticity_threshold).astype(float)
    data['low_elasticity_regime'] = (data['range_elasticity_3d'] <= high_elasticity_threshold).astype(float)
    
    # Elasticity-scaled efficiency momentum
    data['elasticity_scaled_eff_mom'] = data['eff_mom_div'] / (1 + abs(data['range_elasticity_3d']))
    
    # Multi-Timeframe Divergence Enhancement
    data['eff_elasticity_corr'] = (
        data['eff_3d'].rolling(window=15, min_periods=15).corr(data['range_elasticity_3d'])
    )
    data['corr_deviation'] = (
        data['eff_elasticity_corr'] - data['eff_elasticity_corr'].rolling(window=20, min_periods=20).mean()
    )
    
    # Flow-Price Divergence Patterns
    data['positive_divergence'] = (
        (data['volume_accel'] > data['volume_accel'].shift(1)) & 
        (data['eff_mom_div'] < data['eff_mom_div'].shift(1))
    ).astype(float) * abs(data['corr_deviation'])
    
    data['negative_divergence'] = (
        (data['volume_accel'] < data['volume_accel'].shift(1)) & 
        (data['eff_mom_div'] > data['eff_mom_div'].shift(1))
    ).astype(float) * abs(data['corr_deviation'])
    
    # Volume-weighted efficiency metrics
    data['eff_5d_vol_weighted'] = (
        data['eff_5d_abs'] * data['amount_confidence']
    )
    
    data['intraday_eff_amount_conf'] = (
        data['intraday_efficiency'] * data['amount_confidence']
    )
    
    # Dynamic Signal Blending Framework
    # High elasticity component
    high_elasticity_component = (
        data['elasticity_scaled_eff_mom'] * 
        (1 + data['positive_divergence'] - data['negative_divergence']) *
        data['high_elasticity_regime']
    )
    
    # Low elasticity component
    low_elasticity_component = (
        data['intraday_eff_amount_conf'] * 
        data['eff_5d_vol_weighted'] *
        data['low_elasticity_regime']
    )
    
    # Transition regime component (weighted combination)
    transition_weight = abs(data['range_elasticity_3d'] - high_elasticity_threshold) / (
        data['range_elasticity_3d'].rolling(window=20, min_periods=20).std() + 1e-8
    )
    
    transition_component = (
        (high_elasticity_component * transition_weight + 
         low_elasticity_component * (1 - transition_weight)) * 
        (1 - data['high_elasticity_regime'] - data['low_elasticity_regime'])
    )
    
    # Final composite signal
    data['composite_signal'] = (
        high_elasticity_component + 
        low_elasticity_component + 
        transition_component
    )
    
    # Apply volume confirmation and divergence adjustments
    data['final_factor'] = (
        data['composite_signal'] * 
        (1 + 0.5 * data['positive_divergence'] - 0.5 * data['negative_divergence']) *
        data['amount_confidence']
    )
    
    # Apply regime-dependent decay
    decay_factor = np.where(
        data['high_elasticity_regime'] == 1,
        0.9,  # Faster decay in high elasticity
        np.where(
            data['low_elasticity_regime'] == 1,
            0.95,  # Slower decay in low elasticity
            0.92   # Medium decay in transition
        )
    )
    
    # Apply exponential decay to smooth the factor
    data['smoothed_factor'] = data['final_factor'].copy()
    for i in range(1, len(data)):
        if not pd.isna(data['smoothed_factor'].iloc[i-1]):
            data['smoothed_factor'].iloc[i] = (
                decay_factor[i] * data['smoothed_factor'].iloc[i-1] + 
                (1 - decay_factor[i]) * data['final_factor'].iloc[i]
            )
    
    return data['smoothed_factor']
