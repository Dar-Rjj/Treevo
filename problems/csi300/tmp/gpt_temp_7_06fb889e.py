import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Residual Momentum with Volatility-Regime Microstructure Enhancement
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Market benchmark (using average of all stocks as proxy)
    data['market_close'] = data.groupby(level='date')['close'].transform('mean')
    
    # Multi-Timeframe Residual Momentum Calculation
    # Short-Term Residual Momentum (5-day)
    data['stock_return_5d'] = data['close'] / data['close'].shift(5) - 1
    data['market_return_5d'] = data['market_close'] / data['market_close'].shift(5) - 1
    data['residual_5d'] = data['stock_return_5d'] - data['market_return_5d']
    
    # Medium-Term Residual Momentum (20-day)
    data['stock_return_20d'] = data['close'] / data['close'].shift(20) - 1
    data['market_return_20d'] = data['market_close'] / data['market_close'].shift(20) - 1
    data['residual_20d'] = data['stock_return_20d'] - data['market_return_20d']
    
    # Residual Momentum Quality Assessment
    data['momentum_acceleration'] = data['residual_5d'] / (data['residual_20d'] + 1e-8)
    data['directional_consistency'] = np.sign(data['residual_5d']) * np.sign(data['residual_20d'])
    
    # Volatility calculation for weighting
    data['price_volatility_20d'] = data['close'].pct_change().rolling(window=20).std()
    data['volatility_weighted_residual'] = data['residual_5d'] / (data['price_volatility_20d'] + 1e-8)
    
    # Dynamic Volatility-Regime Detection and Analysis
    # True Range calculation
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    
    # Multi-Scale Volatility Environment
    data['volatility_5d_tr'] = data['true_range'].rolling(window=5).mean()
    data['volatility_5d_cc'] = data['close'].pct_change().rolling(window=5).std()
    data['volatility_20d_range'] = (data['high'] - data['low']).rolling(window=20).mean()
    data['volatility_ratio'] = data['volatility_5d_tr'] / (data['volatility_20d_range'] + 1e-8)
    
    # Regime Classification
    data['high_vol_regime'] = (data['volatility_ratio'] > 1.3).astype(int)
    data['low_vol_regime'] = (data['volatility_ratio'] < 0.8).astype(int)
    data['normal_regime'] = ((data['volatility_ratio'] >= 0.8) & (data['volatility_ratio'] <= 1.3)).astype(int)
    
    # Volume-Accelerated Order Flow Analysis
    # Volume Slope Analysis
    def volume_slope(series):
        if len(series) < 5:
            return np.nan
        x = np.arange(len(series))
        return np.polyfit(x, series.values, 1)[0] if not series.isna().any() else np.nan
    
    data['volume_slope_5d'] = data['volume'].rolling(window=5).apply(volume_slope, raw=False)
    data['volume_acceleration'] = data['volume_slope_5d'] / (data['volume'].rolling(window=5).mean() + 1e-8)
    
    # Order Flow Dynamics
    data['signed_order_flow'] = data['volume'] * data['close'].pct_change()
    data['cumulative_order_flow_3d'] = data['signed_order_flow'].rolling(window=3).sum()
    data['order_flow_persistence'] = data['cumulative_order_flow_3d'] / (data['signed_order_flow'].rolling(window=10).std() + 1e-8)
    
    # Regime-Sensitive Factor Construction
    # High Volatility Regime Processing
    data['hv_factor'] = (data['residual_5d'] / (data['volatility_5d_tr'] + 1e-8)) * data['directional_consistency']
    data['hv_volume_filter'] = data['hv_factor'] * np.tanh(data['volume_acceleration'])
    data['hv_order_flow_adj'] = data['hv_volume_filter'] * (1 + np.tanh(data['order_flow_persistence']))
    
    # Low Volatility Regime Processing
    data['lv_factor'] = data['residual_20d'] * data['directional_consistency']
    data['lv_breakout_detection'] = data['lv_factor'] * np.tanh(data['volume_acceleration'] * 2)
    data['price_efficiency'] = abs(data['residual_5d']) / (data['volatility_5d_cc'] + 1e-8)
    data['lv_efficiency_adj'] = data['lv_breakout_detection'] * data['price_efficiency']
    
    # Regime Transition Handling
    regime_weight = 1 / (1 + np.exp(-5 * (data['volatility_ratio'] - 1.0)))
    data['regime_adaptive_factor'] = (
        regime_weight * data['hv_order_flow_adj'] + 
        (1 - regime_weight) * data['lv_efficiency_adj']
    )
    
    # Microstructure-Enhanced Composite Factor
    # Volume-Weighted Residual Acceleration
    data['volume_weighted_acceleration'] = data['momentum_acceleration'] * np.tanh(data['volume_slope_5d'])
    volume_confirmation = np.tanh(data['volume_acceleration'] * data['directional_consistency'])
    data['volume_confirmation_score'] = volume_confirmation
    
    # Order Flow Alignment Enhancement
    of_alignment = np.tanh(data['order_flow_persistence'] * np.sign(data['residual_5d']))
    data['order_flow_alignment'] = of_alignment
    data['cumulative_of_momentum'] = data['cumulative_order_flow_3d'] * data['momentum_acceleration']
    
    # Quality-Weighted Final Factor
    momentum_quality = (
        data['directional_consistency'] * 
        np.tanh(data['momentum_acceleration']) * 
        data['price_efficiency']
    )
    
    # Final composite factor
    data['composite_factor'] = (
        data['regime_adaptive_factor'] * 
        momentum_quality * 
        (1 + data['volume_confirmation_score']) * 
        (1 + data['order_flow_alignment']) * 
        np.tanh(data['cumulative_of_momentum'])
    )
    
    # Cross-sectional normalization
    def cross_sectional_rank(group):
        return group.rank(pct=True) - 0.5
    
    factor = data.groupby(level='date')['composite_factor'].apply(cross_sectional_rank)
    
    return factor
