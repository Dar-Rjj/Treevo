import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum-Efficiency Divergence Factor
    """
    data = df.copy()
    
    # Multi-Timeframe Efficiency Metrics
    # Price Path Efficiency
    data['cum_abs_change_5d'] = data['close'].diff().abs().rolling(window=5).sum()
    data['net_move_5d'] = data['close'] - data['close'].shift(5)
    data['path_efficiency_5d'] = np.where(data['cum_abs_change_5d'] != 0, 
                                         data['net_move_5d'] / data['cum_abs_change_5d'], 0)
    
    data['cum_abs_change_10d'] = data['close'].diff().abs().rolling(window=10).sum()
    data['net_move_10d'] = data['close'] - data['close'].shift(10)
    data['path_efficiency_10d'] = np.where(data['cum_abs_change_10d'] != 0, 
                                          data['net_move_10d'] / data['cum_abs_change_10d'], 0)
    
    # Intraday Efficiency Patterns
    data['daily_efficiency'] = np.where((data['high'] - data['low']) != 0, 
                                       (data['close'] - data['open']) / (data['high'] - data['low']), 0)
    data['efficiency_consistency_3d'] = data['daily_efficiency'].rolling(window=3).std()
    
    # Volume Efficiency Dynamics
    data['cum_abs_volume_change_10d'] = data['volume'].diff().abs().rolling(window=10).sum()
    data['net_volume_move_10d'] = data['volume'] - data['volume'].shift(10)
    data['volume_efficiency_10d'] = np.where(data['cum_abs_volume_change_10d'] != 0, 
                                            data['net_volume_move_10d'] / data['cum_abs_volume_change_10d'], 0)
    
    data['volume_amount_ratio'] = np.where(data['amount'] != 0, data['volume'] / data['amount'], 0)
    data['volume_amount_consistency_5d'] = data['volume_amount_ratio'].rolling(window=5).std()
    
    # Acceleration Regime Components
    # Price Acceleration Hierarchy
    data['momentum_1d'] = data['close'].pct_change(1)
    data['momentum_5d'] = data['close'].pct_change(5)
    data['acceleration_1d'] = data['momentum_1d'].diff()
    data['acceleration_5d'] = data['momentum_5d'].diff()
    
    # Volume Acceleration Structure
    data['volume_momentum_1d'] = data['volume'].pct_change(1)
    data['volume_momentum_5d'] = data['volume'].pct_change(5)
    data['volume_acceleration_1d'] = data['volume_momentum_1d'].diff()
    data['volume_acceleration_5d'] = data['volume_momentum_5d'].diff()
    
    # Efficiency Momentum Tracking
    data['intraday_efficiency_change_3d'] = data['daily_efficiency'].diff(3)
    data['path_efficiency_momentum_5d'] = data['path_efficiency_5d'].diff(5)
    
    # Multi-Dimensional Divergence
    # Efficiency-Acceleration Alignment
    data['efficiency_accel_alignment'] = np.sign(data['path_efficiency_5d']) * np.sign(data['acceleration_5d'])
    data['high_eff_decel'] = (data['path_efficiency_5d'].abs() > data['path_efficiency_5d'].abs().rolling(20).mean()) & (data['acceleration_5d'] < 0)
    data['low_eff_accel'] = (data['path_efficiency_5d'].abs() < data['path_efficiency_5d'].abs().rolling(20).mean()) & (data['acceleration_5d'] > 0)
    
    # Price-Volume Momentum Divergence
    data['price_volume_divergence'] = np.sign(data['acceleration_5d']) * np.sign(data['volume_acceleration_5d'])
    data['pos_price_neg_volume'] = (data['acceleration_5d'] > 0) & (data['volume_acceleration_5d'] < 0)
    data['neg_price_pos_volume'] = (data['acceleration_5d'] < 0) & (data['volume_acceleration_5d'] > 0)
    
    # Volatility-Regime Adjusted Divergence
    data['volatility_20d'] = (data['high'] - data['low']).rolling(window=20).mean()
    volatility_regime = data['volatility_20d'] > data['volatility_20d'].rolling(window=60).mean()
    data['volatility_weight'] = np.where(volatility_regime, 1 / (1 + data['volatility_20d']), 1)
    
    # Regime-Adaptive Signal Construction
    # Efficiency-Convergence Regime
    efficiency_convergence = (data['path_efficiency_5d'].abs() > data['path_efficiency_5d'].abs().rolling(20).mean()) & \
                            (data['efficiency_accel_alignment'] > 0) & \
                            (data['path_efficiency_momentum_5d'] > 0)
    
    # Volatility-Transition Regime
    volatility_shift = data['volatility_20d'].pct_change(5).abs() > data['volatility_20d'].pct_change(5).abs().rolling(20).mean()
    
    # Liquidity-Weighted Signal Adjustment
    volume_rank = data['volume'].rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    data['liquidity_weight'] = volume_rank * data['volume_amount_ratio'] / data['volume_amount_ratio'].rolling(20).mean()
    
    # Final Factor Integration
    divergence_strength = (
        data['efficiency_accel_alignment'] * -1 +  # Negative alignment indicates divergence
        data['price_volume_divergence'] * -1 +     # Negative alignment indicates divergence
        data['high_eff_decel'].astype(int) * 2 +
        data['low_eff_accel'].astype(int) * 2 +
        data['pos_price_neg_volume'].astype(int) * 1.5 +
        data['neg_price_pos_volume'].astype(int) * 1.5
    )
    
    # Apply regime weights and adjustments
    regime_multiplier = np.where(efficiency_convergence, 1.5, 1.0)
    transition_adjustment = np.where(volatility_shift, 0.7, 1.0)
    
    final_factor = (
        divergence_strength * 
        data['volatility_weight'] * 
        regime_multiplier * 
        transition_adjustment * 
        data['liquidity_weight']
    )
    
    # Validate Pattern Persistence
    pattern_consistency = data['daily_efficiency'].rolling(window=5).std() < data['daily_efficiency'].rolling(window=20).std()
    trend_alignment = np.sign(data['path_efficiency_5d']) == np.sign(data['path_efficiency_10d'])
    
    # Final factor with persistence validation
    validated_factor = final_factor * pattern_consistency.astype(int) * trend_alignment.astype(int)
    
    # Normalize and bound the factor
    factor_series = validated_factor.rolling(window=60).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8)
    )
    
    return factor_series
