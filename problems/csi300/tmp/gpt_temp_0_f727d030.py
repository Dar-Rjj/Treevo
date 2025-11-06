import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a composite alpha factor combining price-volume divergence, range efficiency,
    volume-confirmed reversal, amount flow persistence, and volatility-volume regime analysis.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price-Volume Divergence Momentum
    # Multi-timeframe Momentum
    mom_5 = data['close'] / data['close'].shift(5) - 1
    mom_10 = data['close'] / data['close'].shift(10) - 1
    mom_ratio = mom_5 / (mom_10 + 1e-8)  # Avoid division by zero
    
    # Volume Confirmation
    vol_trend = data['volume'] / (data['volume'].shift(5) + 1e-8)
    vol_accel = vol_trend / (data['volume'].shift(5) / (data['volume'].shift(10) + 1e-8) + 1e-8)
    vol_mom_corr = np.sign(mom_5) * vol_trend
    
    # Divergence Detection
    bullish_div = (mom_5 > mom_5.rolling(10).mean()) & (vol_trend < 1)
    bearish_div = (mom_5 < mom_5.rolling(10).mean()) & (vol_trend > 1)
    div_strength = np.abs(mom_5) / (vol_trend + 1e-8)
    
    # Range Efficiency Factor
    # True Range Components
    daily_range = data['high'] - data['low']
    gap_adj_range = np.maximum(data['high'], data['close'].shift(1)) - np.minimum(data['low'], data['close'].shift(1))
    effective_range = np.minimum(daily_range, gap_adj_range)
    
    # Movement Efficiency
    net_movement = np.abs(data['close'] - data['close'].shift(1))
    total_movement = daily_range
    efficiency_ratio = net_movement / (total_movement + 1e-8)
    
    # Multi-period Efficiency
    eff_3d = efficiency_ratio.rolling(3).mean()
    eff_persistence = efficiency_ratio.rolling(5).apply(lambda x: (x > 0.5).sum())
    eff_trend = efficiency_ratio / (efficiency_ratio.shift(3) + 1e-8)
    
    # Volume-Confirmed Extreme Reversal
    # Extreme Move Identification
    price_dev_3d = (data['close'] - data['close'].shift(3)) / (data['high'].rolling(3).max() - data['low'].rolling(3).min() + 1e-8)
    vol_spike = data['volume'] / data['volume'].rolling(5).median()
    extreme_threshold = price_dev_3d > (2 * price_dev_3d.rolling(10).std())
    
    # Reversal Signal
    reversal_signal = extreme_threshold & (vol_spike > 1.5)
    
    # Amount Flow Persistence
    # Directional Flow Analysis
    up_day_mask = data['close'] > data['close'].shift(1)
    down_day_mask = data['close'] < data['close'].shift(1)
    up_day_flow = data['amount'].where(up_day_mask, 0)
    down_day_flow = data['amount'].where(down_day_mask, 0)
    net_flow = up_day_flow - down_day_flow
    
    # Flow Momentum
    flow_3d_sum = net_flow.rolling(3).sum()
    flow_dir_consistency = net_flow.rolling(5).apply(lambda x: (np.sign(x) == np.sign(x.shift(1))).sum())
    flow_accel = net_flow / (net_flow.shift(3) + 1e-8)
    
    # Volatility-Volume Regime Alpha
    # Volatility Regime Classification
    vol_10d = (data['high'].rolling(10).max() - data['low'].rolling(10).min()) / data['close'].shift(10)
    vol_ratio = data['close'].rolling(5).std() / (data['close'].shift(5).rolling(5).std() + 1e-8)
    high_vol_regime = vol_ratio > 1.2
    low_vol_regime = vol_ratio < 0.8
    
    # Volume Pattern Recognition
    vol_persistence = data['volume'].rolling(5).apply(lambda x: (x > x.shift(1)).sum())
    vol_clustering = data['volume'] / data['volume'].rolling(3).mean()
    
    # Regime-Adaptive Signals
    trend_continuation = high_vol_regime & (vol_persistence > 3)
    potential_reversal = low_vol_regime & (vol_spike > 1.5)
    
    # Composite Alpha Factor
    # Combine components with appropriate weights
    alpha = (
        # Price-Volume Divergence (30%)
        0.3 * (mom_ratio + vol_mom_corr + div_strength * (bullish_div.astype(float) - bearish_div.astype(float))) +
        # Range Efficiency (25%)
        0.25 * (efficiency_ratio + eff_3d + eff_persistence) +
        # Volume-Confirmed Reversal (20%)
        0.2 * (reversal_signal.astype(float) * vol_spike) +
        # Amount Flow Persistence (15%)
        0.15 * (flow_3d_sum + flow_dir_consistency + flow_accel) +
        # Volatility-Volume Regime (10%)
        0.1 * (trend_continuation.astype(float) - potential_reversal.astype(float))
    )
    
    return alpha
