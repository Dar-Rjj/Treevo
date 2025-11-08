import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Ensure we have enough data for calculations
    if len(data) < 10:
        return factor
    
    # Momentum Acceleration & Exhaustion components
    # Intraday Momentum Efficiency
    intraday_momentum = (data['close'] - data['open']) / (data['high'] - data['low'])
    intraday_momentum = intraday_momentum.replace([np.inf, -np.inf], np.nan)
    
    # Multi-Day Acceleration
    multi_day_accel = (data['close'] - data['close'].shift(2)) / (data['close'].shift(2) - data['close'].shift(4))
    multi_day_accel = multi_day_accel.replace([np.inf, -np.inf], np.nan)
    
    # Momentum Exhaustion
    momentum_exhaustion = (data['high'] - data['close']) / (data['close'] - data['low'])
    momentum_exhaustion = momentum_exhaustion.replace([np.inf, -np.inf], np.nan)
    
    # Volume-Price Alignment components
    # Volume-Momentum Ratio
    volume_momentum_ratio = data['volume'] / abs(data['close'] - data['close'].shift(1))
    volume_momentum_ratio = volume_momentum_ratio.replace([np.inf, -np.inf], np.nan)
    
    # Volume Spike Persistence
    volume_spike_persistence = pd.Series(0, index=data.index)
    for i in range(4, len(data)):
        count = 0
        for j in range(i-4, i+1):
            if j > 0 and data['volume'].iloc[j] > 1.5 * data['volume'].iloc[j-1]:
                count += 1
        volume_spike_persistence.iloc[i] = count
    
    # Price-Volume Divergence
    price_volume_divergence = np.sign(data['close'] - data['close'].shift(1)) * (data['volume'] / data['volume'].shift(1))
    price_volume_divergence = price_volume_divergence.replace([np.inf, -np.inf], np.nan)
    
    # Price Level Dynamics components
    # Support/Resistance Proximity
    support_resistance = (data['close'] - data['low'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))
    support_resistance = support_resistance.replace([np.inf, -np.inf], np.nan)
    
    # Price Compression Breakout
    price_compression = pd.Series(0, index=data.index)
    for i in range(4, len(data)):
        avg_range = np.mean([data['high'].iloc[j] - data['low'].iloc[j] for j in range(i-4, i)])
        if (data['high'].iloc[i] - data['low'].iloc[i]) < 0.7 * avg_range:
            price_compression.iloc[i] = (data['close'].iloc[i] - data['open'].iloc[i]) / (data['high'].iloc[i] - data['low'].iloc[i])
    
    # Level Break Efficiency
    level_break = pd.Series(0, index=data.index)
    for i in range(1, len(data)):
        if data['close'].iloc[i] > data['high'].iloc[i-1]:
            level_break.iloc[i] = (data['close'].iloc[i] - data['high'].iloc[i-1]) / (data['high'].iloc[i] - data['low'].iloc[i])
    
    # Regime Transition Detection components
    # Momentum Regime Shift
    momentum_regime = pd.Series(0, index=data.index)
    for i in range(9, len(data)):
        std_recent = np.std([data['close'].iloc[j] for j in range(i-4, i+1)])
        std_prior = np.std([data['close'].iloc[j] for j in range(i-9, i-4)])
        if std_prior > 0:
            momentum_regime.iloc[i] = std_recent / std_prior
    
    # Trend Exhaustion
    trend_exhaustion = pd.Series(0, index=data.index)
    for i in range(4, len(data)):
        count = 0
        for j in range(i-4, i+1):
            if j >= 2:
                ret1 = data['close'].iloc[j] - data['close'].iloc[j-1]
                ret2 = data['close'].iloc[j-1] - data['close'].iloc[j-2]
                if ret1 * ret2 < 0:
                    count += 1
        trend_exhaustion.iloc[i] = count
    
    # Transition Smoothness
    transition_smoothness = abs((data['close'] - data['close'].shift(1)) - (data['close'].shift(1) - data['close'].shift(2))) / (data['high'] - data['low'])
    transition_smoothness = transition_smoothness.replace([np.inf, -np.inf], np.nan)
    
    # Volume-Weighted Momentum components
    # Volume-Enhanced Returns
    volume_enhanced_returns = (data['close'] - data['close'].shift(1)) * data['volume']
    
    # Volume-Adjusted Momentum
    volume_adjusted_momentum = pd.Series(0, index=data.index)
    for i in range(4, len(data)):
        volume_sum = sum([data['volume'].iloc[j] for j in range(i-4, i+1)])
        if volume_sum > 0:
            volume_adjusted_momentum.iloc[i] = (data['close'].iloc[i] - data['close'].iloc[i-4]) / volume_sum
    
    # Volume Concentration
    volume_concentration = pd.Series(0, index=data.index)
    for i in range(4, len(data)):
        volume_sum = sum([data['volume'].iloc[j] for j in range(i-4, i+1)])
        if volume_sum > 0:
            volume_concentration.iloc[i] = data['volume'].iloc[i] / volume_sum
    
    # Combine all components into final factor
    # Normalize each component and combine with equal weights
    components = [
        intraday_momentum, multi_day_accel, momentum_exhaustion,
        volume_momentum_ratio, volume_spike_persistence, price_volume_divergence,
        support_resistance, price_compression, level_break,
        momentum_regime, trend_exhaustion, transition_smoothness,
        volume_enhanced_returns, volume_adjusted_momentum, volume_concentration
    ]
    
    # Z-score normalize each component and combine
    normalized_components = []
    for comp in components:
        if comp.notna().any():
            mean_val = comp.mean()
            std_val = comp.std()
            if std_val > 0:
                normalized = (comp - mean_val) / std_val
                normalized_components.append(normalized)
    
    # Combine all normalized components with equal weights
    if normalized_components:
        factor = sum(normalized_components) / len(normalized_components)
    
    return factor
