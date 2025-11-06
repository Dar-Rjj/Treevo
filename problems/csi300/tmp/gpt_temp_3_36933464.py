import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Price Efficiency Score
    # Realized gain efficiency
    daily_potential_gain = (data['high'] - data['open']) / data['open']
    daily_captured_gain = (data['close'] - data['open']) / data['open']
    efficiency_ratio = daily_captured_gain / (daily_potential_gain + 1e-8)
    
    # Volatility-adjusted slippage
    true_range_efficiency = (data['close'] - data['open']) / ((data['high'] - data['low']) + 1e-8)
    
    # Directional persistence (3-day sign consistency)
    close_returns = data['close'].pct_change()
    sign_consistency = (close_returns.rolling(window=3).apply(
        lambda x: 1 if len(x) == 3 and (x > 0).sum() >= 2 else (-1 if len(x) == 3 and (x < 0).sum() >= 2 else 0)
    ))
    
    # Combine efficiency components
    efficiency_score = 0.4 * efficiency_ratio + 0.4 * true_range_efficiency + 0.2 * sign_consistency
    
    # Bidirectional Flow Pressure
    # Calculate daily midpoint
    daily_midpoint = (data['high'] + data['low']) / 2
    
    # Upside pressure quantification
    # High-attraction volume (simplified using close position relative to range)
    high_attraction_ratio = (data['close'] - daily_midpoint) / ((data['high'] - daily_midpoint) + 1e-8)
    high_attraction_volume = data['volume'] * np.maximum(0, high_attraction_ratio)
    
    # Resistance breakthrough
    resistance_breakthrough = (data['close'] > 0.9 * data['high']).astype(int)
    
    # Downside pressure quantification  
    # Low-rejection volume
    low_rejection_ratio = (daily_midpoint - data['close']) / ((daily_midpoint - data['low']) + 1e-8)
    low_rejection_volume = data['volume'] * np.maximum(0, low_rejection_ratio)
    
    # Support defense
    support_defense = (data['close'] > data['low'] * 1.1).astype(int)
    
    # Net flow pressure
    upside_pressure = 0.6 * high_attraction_volume.rolling(5).mean() + 0.4 * resistance_breakthrough.rolling(5).sum()
    downside_pressure = 0.6 * low_rejection_volume.rolling(5).mean() + 0.4 * support_defense.rolling(5).sum()
    net_flow_pressure = (upside_pressure - downside_pressure) / (upside_pressure + downside_pressure + 1e-8)
    
    # Efficiency-Flow Divergence
    # 5-day momentum calculations
    efficiency_momentum = efficiency_score.rolling(5).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) if len(x) == 5 else 0
    )
    flow_momentum = net_flow_pressure.rolling(5).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) if len(x) == 5 else 0
    )
    
    # Regime detection
    bullish_convergence = (efficiency_momentum > 0) & (flow_momentum > 0)
    bearish_divergence = (efficiency_momentum < 0) & (flow_momentum > 0)
    efficiency_flow_divergence = (efficiency_momentum > 0) & (flow_momentum < 0)
    neutral_alignment = ~(bullish_convergence | bearish_divergence | efficiency_flow_divergence)
    
    # Adaptive Signal Generation
    # Base signal (efficiency adjusted by flow pressure)
    base_signal = efficiency_score * (1 + net_flow_pressure)
    
    # Volume intensity
    volume_zscore = (data['volume'] - data['volume'].rolling(20).mean()) / (data['volume'].rolling(20).std() + 1e-8)
    volume_intensity = np.tanh(volume_zscore / 3)  # Normalized to [-1, 1]
    
    # Regime-based weighting
    regime_weights = pd.Series(1.0, index=data.index)
    regime_weights[bullish_convergence] = 1.2 + 0.3 * volume_intensity[bullish_convergence]
    regime_weights[bearish_divergence] = 0.7 - 0.2 * volume_intensity[bearish_divergence]
    regime_weights[efficiency_flow_divergence] = 0.8 + 0.1 * volume_intensity[efficiency_flow_divergence]
    regime_weights[neutral_alignment] = 1.0 + 0.1 * volume_intensity[neutral_alignment]
    
    # Final factor with regime weighting and volume confirmation
    final_factor = base_signal * regime_weights * (1 + 0.2 * volume_intensity)
    
    return final_factor
