import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Volatility-Asymmetric Momentum with Microstructure Confirmation
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Adaptive Momentum Engine
    # Multi-Timeframe Momentum Acceleration
    mom_5d = data['close'] / data['close'].shift(5) - 1
    mom_20d = data['close'] / data['close'].shift(20) - 1
    momentum_acceleration = mom_5d - mom_20d
    
    # Volatility-Regime Momentum Filtering
    # Calculate daily true range
    high_low = data['high'] - data['low']
    high_close_prev = abs(data['high'] - data['close'].shift(1))
    low_close_prev = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Compute volatility momentum
    vol_momentum = true_range.rolling(10).std() / true_range.shift(10).rolling(10).std()
    vol_momentum = vol_momentum.replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Apply volatility-weighted momentum scaling
    volatility_weighted_momentum = momentum_acceleration * vol_momentum
    
    # Price-Volume Asymmetry Divergence System
    # Directional Volume Asymmetry Patterns
    # Up-Day Volume Concentration
    price_change = data['close'].pct_change()
    up_days = price_change > 0
    down_days = price_change < 0
    
    # Up-Day Volume Ratio
    up_volume_avg = data['volume'].rolling(5).apply(
        lambda x: x[up_days.loc[x.index].values].mean() if up_days.loc[x.index].sum() > 0 else 1
    )
    up_volume_ratio = data['volume'] / up_volume_avg.replace(0, 1)
    
    # Down-Day Volume Distribution
    down_volume_avg = data['volume'].rolling(5).apply(
        lambda x: x[down_days.loc[x.index].values].mean() if down_days.loc[x.index].sum() > 0 else 1
    )
    down_volume_intensity = data['volume'] / down_volume_avg.replace(0, 1)
    
    # Volume Asymmetry Index
    volume_asymmetry = up_volume_ratio - down_volume_intensity
    
    # Multi-Scale Divergence Detection
    # Short-Term Price-Volume Mismatch (1-3 days)
    price_trend_3d = data['close'].pct_change(3)
    volume_trend_3d = data['volume'].pct_change(3)
    short_term_divergence = np.sign(price_trend_3d) != np.sign(volume_trend_3d)
    short_term_divergence = short_term_divergence.astype(float) * np.abs(price_trend_3d)
    
    # Medium-Term Divergence Patterns (3-5 days)
    price_trend_5d = data['close'].pct_change(5)
    volume_trend_5d = data['volume'].pct_change(5)
    medium_term_divergence = np.sign(price_trend_5d) != np.sign(volume_trend_5d)
    medium_term_divergence = medium_term_divergence.astype(float) * np.abs(price_trend_5d)
    
    # Cross-Timeframe Divergence Alignment
    multi_scale_divergence = (short_term_divergence.rolling(3).mean() + 
                             medium_term_divergence.rolling(5).mean()) / 2
    
    # Microstructure Asymmetry Confirmation
    # Order Flow Asymmetry Analysis
    directional_order_flow = data['volume'] * (data['close'] - data['close'].shift(1))
    cumulative_order_flow = directional_order_flow.rolling(3).sum()
    
    # Transaction-Size Based Confirmation (using amount as proxy for trade size)
    avg_trade_size = data['amount'] / data['volume'].replace(0, 1)
    large_trade_bias = (avg_trade_size * directional_order_flow).rolling(5).mean()
    
    # Size-Based Asymmetry Confirmation
    microstructure_confirmation = (cumulative_order_flow + large_trade_bias) / 2
    
    # Dynamic Alpha Synthesis Engine
    # Volatility-Asymmetry Integration
    volatility_asymmetry_integration = (volatility_weighted_momentum * 
                                       volume_asymmetry * 
                                       microstructure_confirmation)
    
    # Multi-Scale Factor Convergence
    multi_scale_convergence = (volatility_asymmetry_integration + 
                              multi_scale_divergence * np.sign(volatility_asymmetry_integration))
    
    # Regime-Adaptive Final Factor
    # Use volatility to determine regime
    volatility_regime = true_range.rolling(20).std()
    high_vol_regime = volatility_regime > volatility_regime.rolling(50).quantile(0.7)
    low_vol_regime = volatility_regime < volatility_regime.rolling(50).quantile(0.3)
    
    # Trending regime: Emphasize divergence acceleration
    trending_factor = multi_scale_convergence * (1 + np.abs(multi_scale_divergence))
    
    # Range-bound regime: Focus on symmetry breaks
    range_bound_factor = multi_scale_convergence * (1 + np.abs(volume_asymmetry))
    
    # Generate final volatility-asymmetric momentum factor
    final_factor = np.where(high_vol_regime, trending_factor, 
                           np.where(low_vol_regime, range_bound_factor, multi_scale_convergence))
    
    return pd.Series(final_factor, index=data.index)
