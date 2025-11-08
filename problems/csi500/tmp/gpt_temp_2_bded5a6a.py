import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Asymmetry Detection
    # Calculate price change
    data['price_change'] = data['close'] - data['close'].shift(1)
    
    # Upside vs Downside Volatility
    data['upside_vol'] = np.where(data['price_change'] > 0, data['high'] - data['close'], 0)
    data['downside_vol'] = np.where(data['price_change'] < 0, data['close'] - data['low'], 0)
    
    # 5-day volatility sums
    data['upside_vol_5d'] = data['upside_vol'].rolling(window=5, min_periods=3).sum()
    data['downside_vol_5d'] = data['downside_vol'].rolling(window=5, min_periods=3).sum()
    
    # Volatility Asymmetry Ratio
    data['vol_asymmetry_ratio'] = (data['upside_vol_5d'] - data['downside_vol_5d']) / (data['upside_vol_5d'] + data['downside_vol_5d'] + 1e-8)
    
    # Volume Flow Imbalance Patterns
    # Intraday Volume Distribution
    data['range'] = data['high'] - data['low']
    data['range'] = np.where(data['range'] == 0, 1e-8, data['range'])  # Avoid division by zero
    
    data['high_vol_concentration'] = data['volume'] * (data['high'] - data['close']) / data['range']
    data['low_vol_concentration'] = data['volume'] * (data['close'] - data['low']) / data['range']
    
    # Net Volume Flow
    data['net_vol_flow'] = data['high_vol_concentration'] - data['low_vol_concentration']
    
    # 3-day Volume Flow Trend
    data['vol_flow_trend'] = data['net_vol_flow'].rolling(window=3, min_periods=2).mean()
    
    # Price-Volume Efficiency Context
    # Volume-Weighted Price Efficiency
    data['efficiency_per_vol'] = (data['close'] - data['open']) / (data['volume'] + 1e-8)
    
    # 5-day efficiency-volume correlation
    data['efficiency_vol_corr'] = data['efficiency_per_vol'].rolling(window=5, min_periods=3).corr(data['volume'].rolling(window=5, min_periods=3).mean())
    
    # Volume-Adjusted Range Utilization
    data['range_utilization'] = abs(data['close'] - data['open']) / data['range']
    data['vol_scaled_range_efficiency'] = data['range_utilization'] * data['volume']
    
    # Asymmetry Convergence Detection
    # Volatility-Volume Alignment
    data['vol_vol_alignment'] = np.where(
        (data['upside_vol'] > data['upside_vol'].shift(1)) & (data['high_vol_concentration'] > data['high_vol_concentration'].shift(1)), 1,
        np.where(
            (data['downside_vol'] > data['downside_vol'].shift(1)) & (data['low_vol_concentration'] > data['low_vol_concentration'].shift(1)), -1, 0
        )
    )
    
    # Efficiency-Asymmetry Divergence
    data['efficiency_asymmetry_div'] = np.where(
        (data['efficiency_per_vol'].abs() > data['efficiency_per_vol'].abs().shift(1)) & 
        (data['vol_asymmetry_ratio'] * data['price_change'] < 0), -1,
        np.where(
            (data['efficiency_per_vol'].abs() < data['efficiency_per_vol'].abs().shift(1)) & 
            (data['net_vol_flow'] * data['price_change'] > 0), 1, 0
        )
    )
    
    # Core Asymmetry Signal Construction
    # Volatility Asymmetry Component
    data['vol_asymmetry_component'] = data['vol_asymmetry_ratio'] * data['net_vol_flow']
    
    # Efficiency-Volume Multiplier
    data['efficiency_vol_multiplier'] = data['efficiency_per_vol'] * data['range_utilization']
    data['efficiency_vol_multiplier'] = data['efficiency_vol_multiplier'] * np.sign(data['vol_flow_trend'])
    
    # Combine components
    data['core_signal'] = data['vol_asymmetry_component'] * data['efficiency_vol_multiplier']
    
    # Regime-Adaptive Signal Enhancement
    # Volatility Context Scaling
    data['volatility_magnitude'] = data['range'].rolling(window=5, min_periods=3).mean()
    data['volatility_persistence'] = data['range'].rolling(window=3, min_periods=2).std() / (data['range'].rolling(window=3, min_periods=2).mean() + 1e-8)
    
    # Volume Regime Weighting
    data['vol_asymmetry_strength'] = abs(data['net_vol_flow']).rolling(window=5, min_periods=3).mean()
    data['volume_regime_weight'] = np.where(data['vol_asymmetry_strength'] > data['vol_asymmetry_strength'].shift(1), 1.2, 0.8)
    
    # Pattern Consistency Filter
    data['consecutive_asymmetry'] = data['vol_asymmetry_ratio'].rolling(window=3, min_periods=2).apply(
        lambda x: 1 if all(x > 0) or all(x < 0) else 0.5
    )
    
    # Final factor construction
    data['factor'] = (
        data['core_signal'] * 
        data['volatility_magnitude'] * 
        (1 + data['volatility_persistence']) * 
        data['volume_regime_weight'] * 
        data['consecutive_asymmetry']
    )
    
    # Normalize the factor
    data['factor'] = (data['factor'] - data['factor'].rolling(window=20, min_periods=10).mean()) / (data['factor'].rolling(window=20, min_periods=10).std() + 1e-8)
    
    return data['factor']
