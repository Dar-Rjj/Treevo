import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining price-volume divergence, volatility-scaled efficiency,
    extreme reversal detection, amount flow persistence, and regime-dependent patterns.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price-Volume Divergence Momentum
    # Multi-timeframe Momentum
    momentum_5 = data['close'] / data['close'].shift(5) - 1
    momentum_10 = data['close'] / data['close'].shift(10) - 1
    momentum_ratio = momentum_5 / momentum_10.replace(0, np.nan)
    
    # Volume Confirmation
    volume_trend = data['volume'] / data['volume'].shift(5)
    
    # Calculate volume persistence (count of volume > previous volume)
    volume_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(5, len(data)):
        window = data['volume'].iloc[i-5:i+1]
        persistence = (window > window.shift(1)).sum()
        volume_persistence.iloc[i] = persistence
    
    volume_momentum_corr = np.sign(momentum_5) * volume_trend
    
    # Divergence Detection
    bullish_div = (momentum_5 > momentum_5.rolling(10).mean()) & (volume_trend < 1)
    bearish_div = (momentum_5 < momentum_5.rolling(10).mean()) & (volume_trend > 1)
    divergence_strength = np.abs(momentum_5) / volume_trend.replace(0, np.nan)
    
    # Volatility-Scaled Range Efficiency
    # True Range Components
    daily_range = data['high'] - data['low']
    gap_adjusted_range = np.maximum(data['high'], data['close'].shift(1)) - np.minimum(data['low'], data['close'].shift(1))
    
    # Average True Range
    tr = np.maximum(data['high'] - data['low'], 
                   np.maximum(np.abs(data['high'] - data['close'].shift(1)), 
                             np.abs(data['low'] - data['close'].shift(1))))
    atr_5 = tr.rolling(5).mean()
    
    # Price Efficiency
    single_day_eff = np.abs(data['close'] - data['close'].shift(1)) / daily_range.replace(0, np.nan)
    
    # 3-day cumulative efficiency
    cum_range_3 = daily_range.rolling(3).sum()
    cum_move_3 = np.abs(data['close'] - data['close'].shift(3))
    eff_3_day = cum_move_3 / cum_range_3.replace(0, np.nan)
    
    # Efficiency persistence
    eff_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(5, len(data)):
        window = single_day_eff.iloc[i-5:i+1]
        persistence = (window > 0.5).sum()
        eff_persistence.iloc[i] = persistence
    
    # Volatility Scaling
    range_volatility = daily_range.rolling(10).std()
    eff_vol_ratio = single_day_eff / range_volatility.replace(0, np.nan)
    
    # Volume-Confirmed Extreme Reversal
    # Extreme Move Identification
    price_deviation = (data['close'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan)
    volume_median_10 = data['volume'].rolling(10).median()
    volume_spike = data['volume'] / volume_median_10.replace(0, np.nan)
    
    # Multi-day extreme
    returns_3 = data['close'].pct_change().rolling(3).apply(lambda x: np.max(np.abs(x)), raw=True)
    
    # Amount Flow Persistence
    # Directional Flow Analysis
    up_days = data['close'] > data['close'].shift(1)
    down_days = data['close'] < data['close'].shift(1)
    
    up_flow = data['amount'] * up_days
    down_flow = data['amount'] * down_days
    net_flow = up_flow - down_flow
    
    # Flow Momentum
    flow_3_day = net_flow.rolling(3).sum()
    flow_acceleration = net_flow / net_flow.shift(3).replace(0, np.nan)
    
    # Flow consistency
    flow_consistency = pd.Series(index=data.index, dtype=float)
    for i in range(5, len(data)):
        window = net_flow.iloc[i-5:i+1]
        consistency = (np.sign(window) == np.sign(window.shift(1))).sum()
        flow_consistency.iloc[i] = consistency
    
    # Regime-Dependent Volume Patterns
    # Volatility Regime Classification
    range_vol = (data['high'].rolling(10).max() - data['low'].rolling(10).min()) / data['close'].shift(10)
    vol_ratio = data['close'].pct_change().rolling(5).std() / data['close'].pct_change().shift(5).rolling(5).std().replace(0, np.nan)
    
    # Volume Clustering
    volume_mean_10 = data['volume'].rolling(10).mean()
    volume_regime = data['volume'] / volume_mean_10.replace(0, np.nan)
    
    # Volume persistence
    volume_cluster_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(5, len(data)):
        window = data['volume'].iloc[i-5:i+1]
        mean_vol = volume_mean_10.iloc[i]
        persistence = (window > mean_vol).sum()
        volume_cluster_persistence.iloc[i] = persistence
    
    # Combine all components into final factor
    factor = (
        # Price-Volume Divergence components
        0.2 * momentum_ratio.fillna(0) +
        0.15 * volume_momentum_corr.fillna(0) +
        0.1 * divergence_strength.fillna(0) * np.where(bullish_div, 1, np.where(bearish_div, -1, 0)) +
        
        # Volatility-Scaled Efficiency components
        0.15 * eff_3_day.fillna(0) +
        0.1 * eff_vol_ratio.fillna(0) +
        0.05 * eff_persistence.fillna(0) / 5 +
        
        # Extreme Reversal components
        0.1 * price_deviation.fillna(0) * volume_spike.fillna(0) +
        0.05 * returns_3.fillna(0) * np.sign(price_deviation.fillna(0)) +
        
        # Amount Flow components
        0.05 * flow_3_day.fillna(0) / data['amount'].rolling(10).mean().replace(0, np.nan).fillna(0) +
        0.03 * flow_acceleration.fillna(0) +
        0.02 * flow_consistency.fillna(0) / 5 +
        
        # Regime-Dependent components
        0.05 * volume_regime.fillna(0) * np.where(range_vol > range_vol.rolling(20).median(), 1, -1)
    )
    
    return factor
