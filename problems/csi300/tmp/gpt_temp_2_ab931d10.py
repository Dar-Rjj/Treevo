import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Volatility-Efficiency Momentum with Price-Volume Convergence
    """
    data = df.copy()
    
    # Multi-Timeframe Efficiency-Volatility Analysis
    # Ultra-Short Efficiency (2-day)
    data['true_range_2d'] = data['high'].rolling(window=2).max() - data['low'].rolling(window=2).min()
    data['price_movement_2d'] = abs(data['close'] - data['close'].shift(2))
    data['efficiency_ratio_2d'] = data['price_movement_2d'] / data['true_range_2d']
    
    # Short-Term Volatility (5-day)
    # Directional volatility ratio
    pos_returns = np.maximum(0, data['close'] - data['close'].shift(1))
    neg_returns = np.maximum(0, data['close'].shift(1) - data['close'])
    
    data['up_volatility'] = pos_returns.rolling(window=5).std()
    data['down_volatility'] = neg_returns.rolling(window=5).std()
    data['directional_vol_ratio'] = data['up_volatility'] / data['down_volatility']
    
    # Volatility acceleration
    returns_5d = (data['close'] / data['close'].shift(1) - 1).rolling(window=5).std()
    returns_prev_5d = (data['close'].shift(5) / data['close'].shift(6) - 1).rolling(window=5).std()
    data['volatility_acceleration'] = returns_5d / returns_prev_5d
    
    # Efficiency Acceleration System
    data['efficiency_ratio_5d'] = abs(data['close'] - data['close'].shift(5)) / (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min())
    data['efficiency_acceleration'] = data['efficiency_ratio_5d'] - data['efficiency_ratio_2d']
    
    # Volatility-Regime Consistency
    sign_changes = []
    for i in range(len(data)):
        if i < 4:
            sign_changes.append(np.nan)
            continue
        count = 0
        for j in range(i-4, i+1):
            if j < 2:
                continue
            sign_current = np.sign(data['close'].iloc[j] - data['close'].iloc[j-1])
            sign_prev = np.sign(data['close'].iloc[j-1] - data['close'].iloc[j-2])
            if sign_current == sign_prev:
                count += 1
        sign_changes.append(count)
    
    data['vol_regime_consistency'] = sign_changes
    data['acceleration_pattern'] = np.sign(data['efficiency_acceleration']) * data['vol_regime_consistency']
    
    # Price-Volume Convergence Dynamics
    # Volume-Efficiency Alignment
    data['volume_efficiency_ratio'] = data['volume'] / (data['high'] - data['low'])
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['liquidity_momentum'] = (data['volume'] / data['volume'].shift(1) - 1) * (data['close'] / data['close'].shift(1) - 1)
    
    # Order Flow Convergence
    data['directional_amount'] = data['amount'] * np.sign(data['close'] - data['close'].shift(1))
    data['cumulative_imbalance_5d'] = data['directional_amount'].rolling(window=5).sum()
    
    # Volume Asymmetry Ratio
    up_volume = []
    down_volume = []
    for i in range(len(data)):
        if i < 4:
            up_volume.append(np.nan)
            down_volume.append(np.nan)
            continue
        up_vol = 0
        down_vol = 0
        for j in range(i-4, i+1):
            if data['close'].iloc[j] > data['open'].iloc[j]:
                up_vol += data['volume'].iloc[j]
            elif data['close'].iloc[j] < data['open'].iloc[j]:
                down_vol += data['volume'].iloc[j]
        up_volume.append(up_vol)
        down_volume.append(down_vol)
    
    data['up_volume_5d'] = up_volume
    data['down_volume_5d'] = down_volume
    data['volume_asymmetry_ratio'] = data['up_volume_5d'] / data['down_volume_5d']
    
    # Convergence Signals
    data['efficiency_volume_alignment'] = data['intraday_efficiency'] * data['volume_efficiency_ratio']
    data['momentum_convergence'] = (data['close'] / data['close'].shift(1) - 1) * data['volume_efficiency_ratio']
    data['order_flow_momentum'] = data['cumulative_imbalance_5d'] * data['volume_asymmetry_ratio']
    
    # Multi-Scale Factor Components
    data['efficiency_component'] = data['acceleration_pattern'] * data['efficiency_acceleration']
    data['volatility_component'] = data['directional_vol_ratio'] * data['volatility_acceleration']
    data['convergence_component'] = data['efficiency_volume_alignment'] * data['order_flow_momentum']
    
    # Regime-Adaptive Interactions
    data['volatility_adjusted_efficiency'] = data['efficiency_component'] * data['vol_regime_consistency']
    data['efficiency_volatility_alignment'] = data['momentum_convergence'] * data['directional_vol_ratio']
    data['regime_persistent_flow'] = data['order_flow_momentum'] * data['vol_regime_consistency']
    
    # Volatility-Sensitive Weighting
    high_vol_regime = data['directional_vol_ratio'] > 1.5
    low_vol_regime = data['directional_vol_ratio'] < 0.7
    
    # Dynamic Factor Synthesis
    # Weighted Component Integration
    efficiency_weight = np.where(high_vol_regime, 0.6, np.where(low_vol_regime, 0.2, 0.4))
    volatility_weight = np.where(high_vol_regime, 0.3, np.where(low_vol_regime, 0.2, 0.3))
    convergence_weight = np.where(high_vol_regime, 0.1, np.where(low_vol_regime, 0.6, 0.3))
    
    weighted_components = (
        efficiency_weight * data['volatility_adjusted_efficiency'] +
        volatility_weight * data['efficiency_volatility_alignment'] +
        convergence_weight * data['regime_persistent_flow']
    )
    
    # Scale by efficiency emphasis
    efficiency_emphasis = 1 + abs(data['efficiency_acceleration'])
    scaled_factor = weighted_components * efficiency_emphasis
    
    # Volume Confirmation Layer
    volume_confirmed = scaled_factor * data['liquidity_momentum']
    direction_adjusted = volume_confirmed * data['volume_asymmetry_ratio']
    efficiency_scaled = direction_adjusted * data['volume_efficiency_ratio']
    
    # Position Context Enhancement
    data['daily_range_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    final_factor = efficiency_scaled * data['daily_range_position']
    
    return final_factor
