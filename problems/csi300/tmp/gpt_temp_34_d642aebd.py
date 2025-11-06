import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Efficiency-Weighted Gap Momentum with Microstructure Regime Adaptation
    """
    data = df.copy()
    
    # Multi-Timeframe Gap Efficiency Framework
    # Volatility-Weighted Gap Analysis
    data['prev_close'] = data['close'].shift(1)
    data['overnight_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    
    # Intraday gap with protection against zero denominator
    denominator = np.abs(data['open'] - data['prev_close'])
    denominator = np.where(denominator == 0, 1e-8, denominator)
    data['intraday_gap'] = (data['close'] - data['open']) / denominator
    
    # 10-day Realized Volatility (Average High-Low range)
    data['daily_range'] = data['high'] - data['low']
    data['vol_10d'] = data['daily_range'].rolling(window=10, min_periods=5).mean()
    
    # Volatility-adjusted gaps
    vol_adj = np.where(data['vol_10d'] == 0, 1e-8, data['vol_10d'])
    data['gap_overnight_vol_adj'] = data['overnight_gap'] / vol_adj
    data['gap_intraday_vol_adj'] = data['intraday_gap'] / vol_adj
    
    # Gap Persistence Assessment
    data['gap_momentum_3d'] = (data['gap_overnight_vol_adj'] + 
                              data['gap_overnight_vol_adj'].shift(1) + 
                              data['gap_overnight_vol_adj'].shift(2))
    
    # Gap direction consistency over 5-day window
    gap_direction = np.sign(data['gap_overnight_vol_adj'])
    data['gap_consistency_5d'] = gap_direction.rolling(window=5, min_periods=3).apply(
        lambda x: np.sum(x == x.iloc[-1]) / len(x) if len(x) > 0 else 0, raw=False
    )
    
    # Efficiency-Weighted Gap Enhancement
    # Multi-Scale Efficiency Calculation
    numerator_eff = np.abs(data['close'] - data['open'])
    denominator_eff = np.maximum(data['high'], data['prev_close']) - np.minimum(data['low'], data['prev_close'])
    denominator_eff = np.where(denominator_eff == 0, 1e-8, denominator_eff)
    data['efficiency_ratio'] = numerator_eff / denominator_eff
    
    # 5-day efficiency
    data['close_5d_ago'] = data['close'].shift(5)
    data['range_sum_5d'] = data['daily_range'].rolling(window=5, min_periods=3).sum()
    range_sum_adj = np.where(data['range_sum_5d'] == 0, 1e-8, data['range_sum_5d'])
    data['efficiency_5d'] = np.abs(data['close'] - data['close_5d_ago']) / range_sum_adj
    
    # 10-day efficiency
    data['close_10d_ago'] = data['close'].shift(10)
    data['range_sum_10d'] = data['daily_range'].rolling(window=10, min_periods=5).sum()
    range_sum_10d_adj = np.where(data['range_sum_10d'] == 0, 1e-8, data['range_sum_10d'])
    data['efficiency_10d'] = np.abs(data['close'] - data['close_10d_ago']) / range_sum_10d_adj
    
    data['efficiency_momentum'] = data['efficiency_5d'] - data['efficiency_10d']
    
    # Efficiency-Gap Integration
    data['gap_efficiency_weighted'] = data['gap_overnight_vol_adj'] * data['efficiency_ratio']
    data['gap_persistence_multiplier'] = 1 + data['efficiency_momentum']
    
    # Microstructure-Volume Momentum Confirmation
    # Core Microstructure Signals
    data['price_impact'] = data['daily_range'] / data['volume'].replace(0, 1e-8)
    data['directional_order_flow'] = np.sign(data['close'] - data['open']) * data['volume']
    
    # Order flow persistence
    order_flow_dir = np.sign(data['directional_order_flow'])
    data['order_flow_persistence'] = order_flow_dir.rolling(window=5, min_periods=3).apply(
        lambda x: np.sum(x == x.iloc[-1]) / len(x) if len(x) > 0 else 0, raw=False
    )
    
    # Efficiency-Weighted Order Flow
    data['order_flow_efficiency_weighted'] = data['directional_order_flow'] * data['efficiency_ratio']
    data['order_flow_momentum_adj'] = data['order_flow_efficiency_weighted'] * (1 + data['efficiency_momentum'])
    
    # Volume-Liquidity Integration
    data['liquidity_context'] = data['daily_range'] / data['volume'].replace(0, 1e-8)
    data['volume_concentration'] = data['amount'] / data['daily_range'].replace(0, 1e-8)
    
    # Volume spike intensity
    data['volume_median_20d'] = data['volume'].rolling(window=20, min_periods=10).median()
    vol_median_adj = np.where(data['volume_median_20d'] == 0, 1e-8, data['volume_median_20d'])
    data['volume_spike_intensity'] = data['volume'] / vol_median_adj
    
    # Volatility-Efficiency Cycle Detection
    # Multi-Timeframe Cycle Analysis
    data['efficiency_range_2d'] = data['efficiency_ratio'].rolling(window=5, min_periods=3).apply(
        lambda x: x.max() - x.min() if len(x) > 0 else 0, raw=False
    )
    data['efficiency_volatility'] = data['efficiency_ratio'].rolling(window=10, min_periods=5).std()
    
    data['efficiency_range_5d'] = data['efficiency_ratio'].rolling(window=15, min_periods=8).apply(
        lambda x: x.max() - x.min() if len(x) > 0 else 0, raw=False
    )
    
    # Volatility-Regime Cycle Adaptation
    data['volatility_5d'] = data['close'].rolling(window=5, min_periods=3).std()
    data['volatility_20d'] = data['close'].rolling(window=20, min_periods=10).std()
    vol_20d_adj = np.where(data['volatility_20d'] == 0, 1e-8, data['volatility_20d'])
    data['volatility_regime'] = data['volatility_5d'] / vol_20d_adj
    
    # Cross-Component Quality Integration
    # Gap-Efficiency-Microstructure Alignment
    gap_dir = np.sign(data['gap_overnight_vol_adj'])
    order_flow_dir = np.sign(data['directional_order_flow'])
    efficiency_mom_dir = np.sign(data['efficiency_momentum'])
    
    data['direction_consistency'] = (
        (gap_dir == order_flow_dir).astype(int) + 
        (gap_dir == efficiency_mom_dir).astype(int) + 
        (order_flow_dir == efficiency_mom_dir).astype(int)
    ) / 3.0
    
    # Regime-Adaptive Component Weighting
    # High volatility regime signals
    high_vol_mask = data['volatility_regime'] > 1.0
    low_vol_mask = data['volatility_regime'] <= 1.0
    
    high_efficiency_mask = data['efficiency_ratio'] > data['efficiency_ratio'].rolling(window=20, min_periods=10).median()
    low_efficiency_mask = ~high_efficiency_mask
    
    # Core Gap-Efficiency Momentum
    gap_efficiency_core = (
        data['gap_efficiency_weighted'] * 
        data['gap_persistence_multiplier'] * 
        (1 + data['gap_consistency_5d'])
    )
    
    # Microstructure-Volume Confirmation
    microstructure_confirmation = (
        data['order_flow_momentum_adj'] * 
        (1 / (1 + data['liquidity_context'])) *  # Inverse relationship with liquidity context
        data['volume_spike_intensity']
    )
    
    # Cycle-Regime Enhancement
    cycle_regime_enhancement = (
        (1 + data['efficiency_momentum']) *  # Efficiency cycle phase
        np.where(high_vol_mask, 1.5, 0.8) *  # Volatility regime strength
        (1 + data['direction_consistency'])   # Cross-component alignment
    )
    
    # Dynamic Component Combination with Regime Adaptation
    # High volatility regime: Emphasize gap breakout and volume spikes
    high_vol_component = (
        gap_efficiency_core * 
        np.sqrt(data['volume_spike_intensity']) * 
        (1 + data['efficiency_momentum'])
    )
    
    # Low volatility regime: Emphasize gap persistence and order flow
    low_vol_component = (
        gap_efficiency_core * 
        data['order_flow_persistence'] * 
        (1 - data['efficiency_range_2d'])  # Focus on compression periods
    )
    
    # Efficiency context adjustment
    high_eff_component = gap_efficiency_core * data['order_flow_persistence']
    low_eff_component = gap_efficiency_core * data['volume_concentration']
    
    # Composite Alpha Generation
    raw_alpha = np.zeros(len(data))
    
    # High volatility, high efficiency
    mask1 = high_vol_mask & high_efficiency_mask
    raw_alpha[mask1] = high_vol_component[mask1] * high_eff_component[mask1]
    
    # High volatility, low efficiency
    mask2 = high_vol_mask & low_efficiency_mask
    raw_alpha[mask2] = high_vol_component[mask2] * low_eff_component[mask2]
    
    # Low volatility, high efficiency
    mask3 = low_vol_mask & high_efficiency_mask
    raw_alpha[mask3] = low_vol_component[mask3] * high_eff_component[mask3]
    
    # Low volatility, low efficiency
    mask4 = low_vol_mask & low_efficiency_mask
    raw_alpha[mask4] = low_vol_component[mask4] * low_eff_component[mask4]
    
    # Apply cycle regime enhancement
    raw_alpha = raw_alpha * cycle_regime_enhancement
    
    # Signal Persistence Enhancement
    # 3-day smoothing for noise reduction
    alpha_smoothed = pd.Series(raw_alpha, index=data.index).rolling(window=3, min_periods=2).mean()
    
    # Volatility-adjusted output stability
    vol_stability_adj = 1 / (1 + data['volatility_5d'])
    final_alpha = alpha_smoothed * vol_stability_adj
    
    # Clean up any infinite or NaN values
    final_alpha = final_alpha.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return final_alpha
