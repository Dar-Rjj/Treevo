import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate composite alpha factor combining momentum divergence, efficiency regime analysis,
    volume-flow convergence, volatility-regime adaptation, and regime-aware composites.
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    # Price Momentum Hierarchy
    price_ultra_short = close / close.shift(1) - 1
    price_short = close / close.shift(3) - 1
    price_medium = close / close.shift(10) - 1
    price_long = close / close.shift(20) - 1
    
    # Volume Momentum Hierarchy
    volume_ultra_short = volume / volume.shift(1) - 1
    volume_short = volume / volume.shift(3) - 1
    volume_medium = volume / volume.shift(10) - 1
    volume_long = volume / volume.shift(20) - 1
    
    # Divergence Detection
    acceleration_divergence = (price_short / price_medium) - (volume_short / volume_medium)
    regime_divergence = np.sign(price_short - price_long) * np.sign(volume_short - volume_long)
    
    # Momentum persistence
    momentum_persistence = pd.Series(index=close.index, dtype=float)
    for i in range(5, len(close)):
        window_prices = [np.sign(price_short.iloc[j]) for j in range(i-5, i)]
        window_volumes = [np.sign(volume_short.iloc[j]) for j in range(i-5, i)]
        momentum_persistence.iloc[i] = sum(1 for p, v in zip(window_prices, window_volumes) if p == v)
    
    # Multi-timeframe Efficiency
    def calc_efficiency(window):
        price_changes = close.diff().abs().rolling(window).sum()
        price_ranges = (high - low).rolling(window).sum()
        return price_changes / price_ranges
    
    efficiency_3d = calc_efficiency(3)
    efficiency_10d = calc_efficiency(10)
    efficiency_20d = calc_efficiency(20)
    
    # Efficiency Momentum
    efficiency_acceleration = (efficiency_3d / efficiency_10d) - (efficiency_3d.shift(3) / efficiency_10d.shift(3))
    efficiency_regime = efficiency_3d / efficiency_20d
    efficiency_price_alignment = np.sign(efficiency_regime) * np.sign(close / close.shift(3) - 1)
    
    # Volume Analysis
    volume_trend_3d = volume / volume.shift(3)
    volume_trend_10d = volume / volume.shift(10)
    volume_trend_20d = volume / volume.shift(20)
    volume_volatility = volume.rolling(5).std() / volume.rolling(5).mean()
    
    volume_persistence = pd.Series(index=close.index, dtype=float)
    for i in range(5, len(close)):
        volume_persistence.iloc[i] = sum(1 for j in range(i-5, i) if volume.iloc[j] > volume.iloc[j-1])
    
    # Amount Flow Analysis
    def calc_net_flow(window):
        flow = pd.Series(index=close.index, dtype=float)
        for i in range(window-1, len(close)):
            flow.iloc[i] = sum(amount.iloc[j] * np.sign(close.iloc[j] - close.iloc[j-1]) 
                             for j in range(i-window+1, i+1))
        return flow
    
    flow_3d = calc_net_flow(3)
    flow_10d = calc_net_flow(10)
    flow_20d = calc_net_flow(20)
    
    flow_direction_consistency = pd.Series(index=close.index, dtype=float)
    for i in range(5, len(close)):
        flow_direction_consistency.iloc[i] = sum(1 for j in range(i-5, i) 
                                               if np.sign(flow_3d.iloc[j]) == np.sign(flow_3d.iloc[j-1]))
    
    flow_magnitude_ratio = flow_3d.abs() / flow_20d.abs()
    
    # Convergence Signals
    volume_flow_alignment = np.sign(volume_trend_3d) * np.sign(flow_3d)
    
    # Multi-scale Volatility
    range_volatility_3d = (high.rolling(3).max() - low.rolling(3).min()) / close.shift(3)
    return_volatility = (close / close.shift(1) - 1).rolling(5).std()
    range_volatility_20d = (high.rolling(20).max() - low.rolling(20).min()) / close.shift(20)
    volatility_regime = range_volatility_3d / range_volatility_20d
    
    # Regime-Aware Momentum
    high_vol_momentum = price_short * volatility_regime
    low_vol_momentum = price_medium / volatility_regime
    regime_transition_momentum = (volatility_regime / volatility_regime.shift(5)) * price_long
    
    # Adaptive Volume Patterns
    volume_volatility_correlation = np.sign(volume_ultra_short) * np.sign(volatility_regime - 1)
    high_vol_volume = volume_persistence * volatility_regime
    low_vol_volume = volume_volatility / volatility_regime
    
    # Composite Alpha Signals
    efficient_momentum = price_short * efficiency_regime
    inefficient_reversal = np.sign(price_medium) * (1 - efficiency_regime)
    efficiency_acceleration_momentum = efficiency_acceleration * price_long
    
    volume_confirmed_momentum = price_short * volume_flow_alignment
    flow_persistent_momentum = price_medium * flow_direction_consistency
    volume_volatility_momentum = price_long * volume_volatility_correlation
    
    high_vol_composite = high_vol_momentum * volume_volatility_correlation
    low_vol_composite = low_vol_momentum * efficiency_regime
    transition_composite = regime_transition_momentum * volume_flow_alignment
    
    # Final composite alpha factor
    alpha_factor = (
        efficient_momentum.fillna(0) * 0.15 +
        inefficient_reversal.fillna(0) * 0.12 +
        efficiency_acceleration_momentum.fillna(0) * 0.10 +
        volume_confirmed_momentum.fillna(0) * 0.13 +
        flow_persistent_momentum.fillna(0) * 0.11 +
        volume_volatility_momentum.fillna(0) * 0.09 +
        high_vol_composite.fillna(0) * 0.08 +
        low_vol_composite.fillna(0) * 0.08 +
        transition_composite.fillna(0) * 0.07 +
        acceleration_divergence.fillna(0) * 0.04 +
        regime_divergence.fillna(0) * 0.03
    )
    
    return alpha_factor
