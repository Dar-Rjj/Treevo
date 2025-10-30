import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-scale Fractal Efficiency
    # Micro-scale (2-day)
    data['micro_efficiency'] = (data['close'] - data['close'].shift(2)) / (
        data['high'].rolling(2).max() - data['low'].rolling(2).min())
    
    # Meso-scale (8-day)
    data['meso_efficiency'] = (data['close'] - data['close'].shift(8)) / (
        data['high'].rolling(8).max() - data['low'].rolling(8).min())
    
    # Macro-scale (21-day)
    data['macro_efficiency'] = (data['close'] - data['close'].shift(21)) / (
        data['high'].rolling(21).max() - data['low'].rolling(21).min())
    
    # Efficiency coherence (product of all scales)
    data['efficiency_coherence'] = data['micro_efficiency'] * data['meso_efficiency'] * data['macro_efficiency']
    
    # Intraday Range Quality
    data['daily_range_efficiency'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['opening_gap_efficiency'] = (data['open'] - data['low']) / (data['high'] - data['low'])
    data['range_expansion_momentum'] = (data['high'] - data['low']) / (
        data['high'].shift(3) - data['low'].shift(3))
    
    # Volume-Liquidity Dynamics
    data['volume_acceleration'] = data['volume'] / data['volume'].shift(3)
    data['dollar_flow_intensity'] = data['amount'] / data['amount'].shift(3)
    data['regime_intensity'] = data['volume_acceleration'] * data['dollar_flow_intensity']
    
    # Momentum Quality Assessment
    data['momentum_alignment'] = np.sign(data['close'].pct_change(3)) * np.sign(data['close'].pct_change(8))
    data['momentum_acceleration'] = (data['close'] - data['close'].shift(3)) / (
        data['close'].shift(3) - data['close'].shift(8))
    
    # Volume-weighted momentum (5-day return × volume trend persistence)
    vol_trend_persistence = data['volume'].rolling(5).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) == 5 and not np.isnan(x).any() else 0
    )
    data['volume_weighted_momentum'] = data['close'].pct_change(5) * vol_trend_persistence
    
    # Regime-Adaptive Signal Construction
    high_intensity_signal = data['efficiency_coherence'] * data['regime_intensity']
    low_intensity_signal = data['daily_range_efficiency'] * (1 - data['regime_intensity'])
    transition_signal = data['momentum_acceleration'] * data['volume_acceleration']
    
    # Combine regime signals (weighted by regime intensity)
    regime_adaptive_signal = (
        data['regime_intensity'] * high_intensity_signal + 
        (1 - data['regime_intensity']) * low_intensity_signal + 
        transition_signal
    )
    
    # Multi-Scale Validation
    efficiency_consistency = data['micro_efficiency'] * data['meso_efficiency'] * data['macro_efficiency']
    volume_momentum_coupling = data['volume_weighted_momentum'] * data['momentum_acceleration']
    range_momentum_alignment = data['range_expansion_momentum'] * data['daily_range_efficiency']
    
    multi_scale_validation = efficiency_consistency * volume_momentum_coupling * range_momentum_alignment
    
    # Final Alpha Synthesis
    volume_weighting = data['volume'] / data['volume'].rolling(8).mean()
    range_positioning = 1 - np.abs(data['daily_range_efficiency'] - 0.5)
    
    # Core alpha factor
    alpha = (
        regime_adaptive_signal * 
        multi_scale_validation * 
        volume_weighting * 
        range_positioning
    )
    
    return alpha
