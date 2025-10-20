import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Price Efficiency Ratio
    # Short-term: 5-day return / 5-day absolute return sum
    ret_5d = data['close'].pct_change(5)
    abs_ret_5d = data['close'].pct_change().abs().rolling(window=5).sum()
    efficiency_short = ret_5d / abs_ret_5d
    
    # Medium-term: 20-day return / 20-day absolute return sum
    ret_20d = data['close'].pct_change(20)
    abs_ret_20d = data['close'].pct_change().abs().rolling(window=20).sum()
    efficiency_medium = ret_20d / abs_ret_20d
    
    # Long-term: 60-day return / 60-day absolute return sum
    ret_60d = data['close'].pct_change(60)
    abs_ret_60d = data['close'].pct_change().abs().rolling(window=60).sum()
    efficiency_long = ret_60d / abs_ret_60d
    
    # Volume-Weighted Efficiency
    # Volume acceleration: 5-day volume / 20-day volume
    vol_5d = data['volume'].rolling(window=5).mean()
    vol_20d = data['volume'].rolling(window=20).mean()
    volume_acceleration = vol_5d / vol_20d
    
    # Efficiency-volume correlation: sign(efficiency) × volume_acceleration
    efficiency_volume_corr = np.sign(efficiency_medium) * volume_acceleration
    
    # Regime-Dependent Momentum
    # Volatility regime: 20-day std dev vs 60-day std dev
    vol_20d = data['close'].pct_change().rolling(window=20).std()
    vol_60d = data['close'].pct_change().rolling(window=60).std()
    volatility_ratio = vol_20d / vol_60d
    
    # High vol momentum: 5-day return × volatility ratio
    high_vol_momentum = ret_5d * volatility_ratio
    
    # Range compression for low vol momentum
    range_20d = (data['high'] - data['low']).rolling(window=20).mean()
    range_60d = (data['high'] - data['low']).rolling(window=60).mean()
    range_compression = range_20d / range_60d
    
    # Low vol momentum: 20-day return × range compression
    low_vol_momentum = ret_20d * range_compression
    
    # Liquidity Absorption
    # Volume-price divergence: price vs volume direction
    price_direction = np.sign(data['close'].pct_change())
    volume_direction = np.sign(data['volume'].pct_change())
    volume_price_divergence = price_direction * volume_direction
    
    # Absorption ratio: (buy pressure - sell pressure) / total_volume
    # Using close vs open as proxy for buy/sell pressure
    daily_range = data['high'] - data['low']
    buy_pressure = np.where(data['close'] > data['open'], 
                           (data['close'] - data['open']) / daily_range, 0)
    sell_pressure = np.where(data['close'] < data['open'], 
                            (data['open'] - data['close']) / daily_range, 0)
    absorption_ratio = (buy_pressure - sell_pressure) / (data['volume'] + 1e-8)
    
    # Range Dynamics
    # Range efficiency: (close - open) / (high - low)
    daily_range = data['high'] - data['low']
    range_efficiency = (data['close'] - data['open']) / (daily_range + 1e-8)
    
    # Range expansion: current range > 5-day average range
    avg_range_5d = daily_range.rolling(window=5).mean()
    range_expansion = (daily_range > avg_range_5d).astype(float)
    
    # Signal Convergence
    # Efficiency-momentum alignment: efficiency × momentum
    efficiency_momentum_alignment = efficiency_medium * ret_5d
    
    # Volume-flow confirmation: volume_acceleration × efficiency_sign
    volume_flow_confirmation = volume_acceleration * np.sign(efficiency_medium)
    
    # Combine all components with appropriate weights
    factor = (
        0.3 * efficiency_short.fillna(0) +
        0.4 * efficiency_medium.fillna(0) +
        0.3 * efficiency_long.fillna(0) +
        0.2 * efficiency_volume_corr.fillna(0) +
        0.25 * np.where(volatility_ratio > 1, high_vol_momentum, low_vol_momentum).fillna(0) +
        0.15 * volume_price_divergence.fillna(0) +
        0.1 * absorption_ratio.fillna(0) +
        0.2 * range_efficiency.fillna(0) +
        0.15 * range_expansion.fillna(0) +
        0.25 * efficiency_momentum_alignment.fillna(0) +
        0.2 * volume_flow_confirmation.fillna(0)
    )
    
    return factor
