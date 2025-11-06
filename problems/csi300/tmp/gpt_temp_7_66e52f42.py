import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum-Elasticity Regime Framework with Volume-Range Confirmation
    """
    data = df.copy()
    
    # Volatility-Regime Momentum Structure
    # Regime Classification
    returns = data['close'] / data['close'].shift(1) - 1
    vol_ratio_5 = returns.rolling(window=5, min_periods=5).std()
    vol_ratio_20 = returns.rolling(window=20, min_periods=20).std()
    volatility_ratio = vol_ratio_5 / vol_ratio_20
    
    high_vol_regime = volatility_ratio > 1.2
    low_vol_regime = volatility_ratio < 0.8
    transition_regime = ~high_vol_regime & ~low_vol_regime
    
    # Regime-Adaptive Momentum Components
    short_term_momentum = data['close'] / data['close'].shift(1) - 1
    medium_term_momentum = data['close'] / data['close'].shift(5) - 1
    momentum_acceleration = (data['close'] / data['close'].shift(1)) / (data['close'].shift(1) / data['close'].shift(2))
    
    # Regime-Specific Momentum Weighting
    high_vol_weight = 0.7 * momentum_acceleration + 0.3 * medium_term_momentum
    low_vol_weight = 0.3 * momentum_acceleration + 0.7 * short_term_momentum
    transition_weight = 0.5 * momentum_acceleration + 0.5 * (short_term_momentum + medium_term_momentum) / 2
    
    regime_momentum = pd.Series(index=data.index, dtype=float)
    regime_momentum[high_vol_regime] = high_vol_weight[high_vol_regime]
    regime_momentum[low_vol_regime] = low_vol_weight[low_vol_regime]
    regime_momentum[transition_regime] = transition_weight[transition_regime]
    
    # Price Elasticity-Range Dynamics
    # Elasticity Components
    resistance_elasticity = (data['high'] - data['close']) / (data['high'] - data['low']).replace(0, np.nan)
    support_elasticity = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    elasticity_ratio = resistance_elasticity / support_elasticity.replace(0, np.nan)
    
    # Range Expansion Metrics
    true_range = pd.DataFrame({
        'hl': data['high'] - data['low'],
        'hc': abs(data['high'] - data['close'].shift(1)),
        'lc': abs(data['low'] - data['close'].shift(1))
    }).max(axis=1)
    
    range_momentum = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan)
    range_acceleration = range_momentum / range_momentum.shift(1).replace(0, np.nan)
    
    # Elasticity-Range Integration
    elasticity_range_alignment = elasticity_ratio * range_momentum
    range_breakout_elasticity = range_acceleration * (1 - abs(elasticity_ratio - 1))
    support_resistance_efficiency = (support_elasticity - resistance_elasticity) * true_range
    
    # Volume-Liquidity Confirmation
    # Volume Momentum Structure
    volume_momentum = data['volume'] / data['volume'].shift(1) - 1
    volume_acceleration = (data['volume'] / data['volume'].shift(1)) / (data['volume'].shift(1) / data['volume'].shift(2))
    
    volume_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(4, len(data)):
        window = data['volume'].iloc[i-4:i+1]
        volume_persistence.iloc[i] = (window > window.shift(1)).sum()
    
    # Liquidity Efficiency Metrics
    amount_efficiency = data['amount'] / (data['high'] - data['low']).replace(0, np.nan)
    price_volume_efficiency = abs(data['close'] - data['close'].shift(1)) / data['amount'].replace(0, np.nan)
    volume_range_efficiency = data['volume'] / true_range.replace(0, np.nan)
    
    # Volume-Elasticity Alignment
    volume_elasticity_convergence = np.sign(volume_momentum) * elasticity_ratio
    efficiency_elasticity_alignment = amount_efficiency * (1 - abs(elasticity_ratio - 1))
    volume_persistence_elasticity = volume_persistence * support_elasticity
    
    # Gap Dynamics Integration
    # Overnight Gap Analysis
    overnight_gap = data['open'] / data['close'].shift(1) - 1
    gap_persistence = (np.sign(overnight_gap) == np.sign(overnight_gap.shift(1))).astype(float)
    gap_magnitude = abs(overnight_gap)
    
    # Intraday Gap Dynamics
    intraday_gap_filling = (data['close'] - data['open']) / abs(data['open'] - data['close'].shift(1)).replace(0, np.nan)
    gap_direction_alignment = np.sign(overnight_gap) * np.sign(data['close'] - data['open'])
    gap_efficiency = abs(data['close'] - data['open']) / true_range.replace(0, np.nan)
    
    # Gap-Momentum Integration
    gap_momentum_alignment = overnight_gap * short_term_momentum
    gap_acceleration = gap_magnitude / gap_magnitude.shift(1).replace(0, np.nan)
    gap_elasticity_interaction = gap_direction_alignment * elasticity_ratio
    
    # Multi-Timeframe Convergence
    # Momentum Hierarchy
    short_term_convergence = (np.sign(short_term_momentum) == np.sign(medium_term_momentum)).astype(float)
    acceleration_consistency = (np.sign(momentum_acceleration) == np.sign(short_term_momentum)).astype(float)
    multi_timeframe_strength = abs(short_term_momentum) * abs(medium_term_momentum)
    
    # Range-Volume Timeframe Alignment
    range_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(4, len(data)):
        window = range_momentum.iloc[i-4:i+1]
        range_persistence.iloc[i] = (window > 1).sum()
    
    volume_range_correlation = np.sign(range_momentum) * np.sign(volume_momentum)
    multi_timeframe_efficiency = amount_efficiency * (1 + range_persistence / 5)
    
    # Elasticity-Timeframe Integration
    elasticity_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(4, len(data)):
        window = elasticity_ratio.iloc[i-4:i+1]
        elasticity_persistence.iloc[i] = (window > 1).sum() - (window < 1).sum()
    
    multi_timeframe_elasticity = elasticity_ratio * multi_timeframe_strength
    timeframe_elasticity_alignment = np.sign(elasticity_persistence) * np.sign(multi_timeframe_strength)
    
    # Regime-Adaptive Signal Construction
    # Core Factor Assembly
    high_vol_core = regime_momentum * elasticity_range_alignment * volume_elasticity_convergence
    low_vol_core = regime_momentum * support_resistance_efficiency * efficiency_elasticity_alignment
    transition_core = regime_momentum * range_breakout_elasticity * gap_momentum_alignment
    
    # Confirmation Framework
    volume_confirmation = volume_persistence * volume_range_correlation
    range_confirmation = range_persistence * (1 - abs(elasticity_ratio - 1))
    timeframe_confirmation = short_term_convergence * timeframe_elasticity_alignment
    
    # Risk Adjustment
    volatility_adjustment = 1 / true_range.replace(0, np.nan)
    gap_risk = 1 / (1 + gap_magnitude)
    efficiency_risk = 1 / (1 + abs(price_volume_efficiency - volume_range_efficiency))
    
    # Final Alpha Generation
    # Regime Selection
    selected_core = pd.Series(index=data.index, dtype=float)
    selected_core[high_vol_regime] = high_vol_core[high_vol_regime]
    selected_core[low_vol_regime] = low_vol_core[low_vol_regime]
    selected_core[transition_regime] = transition_core[transition_regime]
    
    selected_confirmation = volume_confirmation * range_confirmation * timeframe_confirmation
    
    # Signal Combination
    base_signal = selected_core * selected_confirmation
    risk_multiplier = volatility_adjustment * gap_risk * efficiency_risk
    
    # Final Alpha
    final_alpha = base_signal * risk_multiplier
    
    return final_alpha
