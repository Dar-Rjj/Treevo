import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-timeframe Volatility Assessment
    data['high_low_range'] = data['high'] - data['low']
    
    # Micro-volatility (3-day)
    data['micro_vol'] = data['high_low_range'].rolling(window=3).std() / (data['high_low_range'].rolling(window=3).mean() + 1e-8)
    
    # Short-volatility (8-day)
    data['short_vol'] = data['high_low_range'].rolling(window=8).std() / (data['high_low_range'].rolling(window=8).mean() + 1e-8)
    
    # Volatility regime ratio and classification
    data['vol_regime_ratio'] = data['micro_vol'] / (data['short_vol'] + 1e-8)
    data['high_vol_regime'] = (data['vol_regime_ratio'] > 1).astype(int)
    
    # Multi-Scale Momentum Dynamics
    # Micro-scale (3-day)
    data['price_accel_micro'] = (data['close'] / data['close'].shift(1) - 1) - (data['close'].shift(1) / data['close'].shift(2) - 1)
    data['range_efficiency'] = (data['close'] - data['open']) / (data['high_low_range'] + 1e-8)
    data['gap_momentum'] = np.sign(data['close'] - data['open']) * np.abs(data['close'] - data['open']) / (data['high_low_range'] + 1e-8)
    
    # Meso-scale (8-day)
    data['close_ret'] = data['close'] / data['close'].shift(1) - 1
    data['up_days'] = (data['close_ret'] > 0).rolling(window=8).sum()
    data['down_days'] = (data['close_ret'] < 0).rolling(window=8).sum()
    data['directional_persistence'] = data['up_days'] - data['down_days']
    
    # Price efficiency meso
    data['price_change_sum'] = np.abs(data['close'] - data['close'].shift(1)).rolling(window=8).sum()
    data['price_efficiency_meso'] = (data['close'] - data['close'].shift(8)) / (data['price_change_sum'] + 1e-8)
    
    # Range utilization
    data['low_8'] = data['low'].rolling(window=8).min()
    data['high_8'] = data['high'].rolling(window=8).max()
    data['range_utilization'] = (data['close'] - data['low_8']) / (data['high_8'] - data['low_8'] + 1e-8)
    
    # Macro-scale (21-day)
    data['price_change_sum_21'] = np.abs(data['close'] - data['close'].shift(1)).rolling(window=21).sum()
    data['trend_durability'] = (data['close'] - data['close'].shift(21)) / (data['price_change_sum_21'] + 1e-8)
    
    data['vol_clustering'] = data['high_low_range'].rolling(window=21).std() / (data['high_low_range'].rolling(window=21).mean() + 1e-8)
    
    # Price memory (autocorrelation)
    def rolling_autocorr(x):
        if len(x) < 21:
            return np.nan
        return pd.Series(x).autocorr(lag=1)
    
    data['price_memory'] = data['close_ret'].rolling(window=21).apply(rolling_autocorr, raw=False)
    
    # Flow Asymmetry & Acceleration
    # Volume-price divergence
    data['vol_price_div_3d'] = (data['volume'] / data['volume'].shift(3)) - (data['close'] / data['close'].shift(3))
    data['vol_price_div_8d'] = (data['volume'] / data['volume'].shift(8)) - (data['close'] / data['close'].shift(8))
    
    # Volume acceleration dynamics
    data['vol_accel_micro'] = (data['volume'] / data['volume'].shift(2) - 1) - (data['volume'].shift(1) / data['volume'].shift(3) - 1)
    data['vol_accel_short'] = (data['volume'] / data['volume'].shift(5) - 1) - (data['volume'].shift(3) / data['volume'].shift(8) - 1)
    
    # Asymmetric flow impact
    data['volume_median_10'] = data['volume'].rolling(window=10).median()
    data['high_vol_small_move'] = (data['volume'] / data['volume_median_10']) / (np.abs(data['close_ret']) + 0.001)
    data['flow_efficiency'] = np.sign(data['close_ret']) * (data['volume'] / data['volume_median_10'])
    
    # Efficiency-Convexity Framework
    data['price_efficiency'] = (data['close'] - data['open']) / (data['high_low_range'] + 1e-8)
    data['volume_efficiency'] = data['volume'] / (data['high_low_range'] + 1e-8)
    data['price_convexity'] = data['price_accel_micro'] / ((data['high_low_range'] / data['close']) + 1e-8)
    
    # Regime-Adaptive Momentum-Flow Integration
    # Acceleration divergence analysis
    data['accel_div_micro'] = data['price_accel_micro'] - data['vol_accel_micro']
    data['accel_div_short'] = data['price_accel_micro'] - data['vol_accel_short']
    
    # Regime-weighted divergence
    data['regime_weighted_div'] = np.where(
        data['high_vol_regime'] == 1,
        0.6 * data['accel_div_micro'] + 0.4 * data['accel_div_short'],
        0.4 * data['accel_div_micro'] + 0.6 * data['accel_div_short']
    )
    
    # Multi-scale momentum alignment
    micro_momentum = data['gap_momentum'].fillna(0)
    meso_momentum = data['directional_persistence'].fillna(0)
    macro_momentum = data['trend_durability'].fillna(0)
    
    data['micro_meso_alignment'] = np.sign(micro_momentum) * np.sign(meso_momentum)
    data['meso_macro_alignment'] = np.sign(meso_momentum) * np.sign(macro_momentum)
    data['momentum_convergence'] = (np.sign(micro_momentum) * np.sign(meso_momentum) * np.sign(macro_momentum)).fillna(0)
    
    # Volume Synchronization Dynamics
    data['directional_volume_flow'] = data['price_efficiency'] * data['volume']
    data['volume_flow_momentum'] = data['directional_volume_flow'] / (data['directional_volume_flow'].shift(5) + 1e-8) - 1
    
    # Flow persistence
    vol_sign = np.sign(data['volume'] / data['volume'].shift(1) - 1)
    price_sign = np.sign(data['close_ret'])
    same_sign = (vol_sign == price_sign).astype(int)
    data['flow_persistence'] = same_sign.rolling(window=5).sum()
    
    # Composite Alpha Construction
    # Base momentum-flow divergence
    base_divergence = data['regime_weighted_div'] * data['momentum_convergence']
    
    # Volume synchronization enhancement
    volume_sync_enhance = data['volume_flow_momentum'] * data['flow_persistence'] / 5.0
    
    # Efficiency-convexity enhancement
    efficiency_multiplier = data['price_efficiency'] * data['volume_efficiency']
    convexity_adjustment = np.tanh(data['price_convexity'] * 10)  # Scale and bound
    
    # Final alpha factor
    alpha = (
        base_divergence * 0.4 +
        volume_sync_enhance * 0.3 +
        efficiency_multiplier * 0.2 +
        convexity_adjustment * 0.1
    )
    
    # Apply regime-specific scaling
    alpha = np.where(
        data['high_vol_regime'] == 1,
        alpha * 0.8,  # Reduce magnitude in high volatility
        alpha * 1.2   # Enhance in low volatility
    )
    
    return alpha
