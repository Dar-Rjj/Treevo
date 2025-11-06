import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate market averages (using rolling window for historical data only)
    market_avg_hl_ratio = (df['high'] - df['low']).div(df['close']).rolling(window=20, min_periods=10).mean()
    market_avg_volume_ratio = df['volume'].div(df['volume'].rolling(window=5, min_periods=3).mean()).rolling(window=20, min_periods=10).mean()
    market_avg_co_hl_ratio = (df['close'] - df['open']).div(df['high'] - df['low']).replace([np.inf, -np.inf], np.nan).rolling(window=20, min_periods=10).mean()
    market_avg_volume = df['volume'].rolling(window=20, min_periods=10).mean()
    market_avg_open_close = (df['open'] - df['close'].shift(1)).rolling(window=20, min_periods=10).mean()
    market_avg_close_open = (df['close'] - df['open']).rolling(window=20, min_periods=10).mean()
    market_avg_price_leadership = pd.Series(index=df.index, dtype=float)
    
    # Cross-Asset Microstructure Contagion Detection
    price_impact_divergence = abs((df['high'] - df['low']).div(df['close']) - market_avg_hl_ratio)
    volume_velocity_divergence = df['volume'].div(df['volume'].rolling(window=5, min_periods=3).mean()) - market_avg_volume_ratio
    spread_efficiency_divergence = (df['close'] - df['open']).div(df['high'] - df['low']).replace([np.inf, -np.inf], np.nan) - market_avg_co_hl_ratio
    
    # Calculate cross-asset pressure using rolling window
    cross_asset_pressure = pd.concat([price_impact_divergence, volume_velocity_divergence, spread_efficiency_divergence], axis=1).abs().sum(axis=1).div(3)
    
    volume_contagion_score = volume_velocity_divergence * cross_asset_pressure
    
    # Rank calculations
    microstructure_contagion_index = cross_asset_pressure.rolling(window=10, min_periods=5).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]) * \
                                   volume_contagion_score.rolling(window=10, min_periods=5).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Liquidity Spillover Network Analysis
    market_return = df['close'].pct_change()
    incoming_liquidity_pressure = market_avg_volume.where(market_return > 0, 0)
    outgoing_liquidity_pressure = market_avg_volume.where(market_return < 0, 0)
    net_liquidity_flow = incoming_liquidity_pressure - outgoing_liquidity_pressure
    liquidity_spillover_ratio = net_liquidity_flow.div(incoming_liquidity_pressure + outgoing_liquidity_pressure).replace([np.inf, -np.inf], 0)
    
    # Liquidity Cluster Detection
    volume_cluster_intensity = df['volume'].rolling(window=5, min_periods=3).max().div(market_avg_volume.rolling(window=5, min_periods=3).max())
    cross_asset_volume_correlation = df['volume'].rolling(window=10, min_periods=5).corr(market_avg_volume)
    liquidity_cluster_score = volume_cluster_intensity * cross_asset_volume_correlation
    
    # Price Discovery Efficiency with Cross-Asset Validation
    opening_discovery_efficiency = (df['open'] - df['close'].shift(1)).div(market_avg_open_close)
    closing_discovery_efficiency = (df['close'] - df['open']).div(market_avg_close_open)
    
    # Calculate correlation safely
    intraday_discovery_consistency = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 4:
            window_data = pd.concat([opening_discovery_efficiency.iloc[i-4:i+1], 
                                   closing_discovery_efficiency.iloc[i-4:i+1]], axis=1)
            if not window_data.isna().any().any():
                intraday_discovery_consistency.iloc[i] = window_data.iloc[:, 0].corr(window_data.iloc[:, 1])
    
    price_leadership_score = (opening_discovery_efficiency + closing_discovery_efficiency).div(2)
    market_avg_price_leadership = price_leadership_score.rolling(window=20, min_periods=10).mean()
    validation_strength = abs(price_leadership_score - market_avg_price_leadership)
    discovery_efficiency_factor = price_leadership_score * validation_strength
    
    # Microstructure Regime Transition Dynamics
    atr_3 = (df['high'] - df['low']).rolling(window=3, min_periods=2).mean()
    atr_10 = (df['high'] - df['low']).rolling(window=10, min_periods=5).mean()
    volatility_regime_change = atr_3.div(df['close']) - atr_10.div(df['close'])
    
    volume_regime_change = df['volume'].div(df['volume'].rolling(window=10, min_periods=5).mean()) - \
                          df['volume'].rolling(window=5, min_periods=3).mean().div(df['volume'].rolling(window=10, min_periods=5).mean())
    
    regime_transition_score = volatility_regime_change * volume_regime_change
    
    pre_transition_momentum = (df['close'] - df['close'].rolling(window=3, min_periods=2).mean()).div(df['close'])
    post_transition_momentum = (df['close'] - df['close'].shift(1)).div(df['close'])
    transition_momentum_gap = post_transition_momentum - pre_transition_momentum
    
    # Cross-Asset Factor Integration
    base_contagion_signal = microstructure_contagion_index * liquidity_spillover_ratio
    enhanced_momentum = base_contagion_signal * transition_momentum_gap
    
    liquidity_weighted_efficiency = discovery_efficiency_factor * liquidity_cluster_score
    weighted_discovery_signal = liquidity_weighted_efficiency * intraday_discovery_consistency
    
    # Signal validation correlation
    signal_validation = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 4:
            window_data = pd.concat([enhanced_momentum.iloc[i-4:i+1], 
                                   weighted_discovery_signal.iloc[i-4:i+1]], axis=1)
            if not window_data.isna().any().any():
                signal_validation.iloc[i] = window_data.iloc[:, 0].corr(window_data.iloc[:, 1])
    
    validation_adjusted_signal = enhanced_momentum * signal_validation
    final_cross_asset_alpha = validation_adjusted_signal * weighted_discovery_signal
    
    # Dynamic Factor Allocation
    # Regime classification
    contagion_threshold = cross_asset_pressure.rolling(window=20, min_periods=10).quantile(0.7)
    regime_based_weight = pd.Series(index=df.index, dtype=float)
    
    high_contagion_mask = cross_asset_pressure > contagion_threshold
    low_contagion_mask = cross_asset_pressure <= contagion_threshold.rolling(window=20, min_periods=10).quantile(0.3)
    transition_mask = ~high_contagion_mask & ~low_contagion_mask
    
    regime_based_weight[high_contagion_mask] = microstructure_contagion_index[high_contagion_mask]
    regime_based_weight[low_contagion_mask] = liquidity_cluster_score[low_contagion_mask]
    regime_based_weight[transition_mask] = regime_transition_score[transition_mask]
    
    # Signal Quality Assessment
    signal_persistence = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 4:
            window = final_cross_asset_alpha.iloc[max(0, i-4):i+1]
            signal_persistence.iloc[i] = (window > 0).sum()
    
    signal_magnitude = abs(final_cross_asset_alpha)
    signal_quality = signal_persistence * signal_magnitude
    
    # Adaptive Alpha Output
    quality_weighted_signal = final_cross_asset_alpha * signal_quality
    regime_adaptive_alpha = quality_weighted_signal * regime_based_weight
    final_alpha = regime_adaptive_alpha * cross_asset_pressure
    
    return final_alpha.fillna(0)
