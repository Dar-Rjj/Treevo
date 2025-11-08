import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum Regime Detection
    # Acceleration-Deceleration Pattern
    accel_pattern = (df['close'] / df['close'].shift(3) - 1) - (df['close'] / df['close'].shift(8) - 1)
    
    # Reversal Probability Score
    reversal_prob = (df['high'] - df['close']) / (df['close'] - df['low'])
    reversal_prob = reversal_prob.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Momentum Transition Signal
    momentum_transition = accel_pattern * reversal_prob
    
    # Volume Structure Asymmetry Analysis
    # Volume Concentration Index
    volume_concentration = df['volume'] / (df['amount'] / (df['high'] - df['low']))
    volume_concentration = volume_concentration.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Volume Flow Direction
    volume_flow = (df['close'] - df['open']) * df['volume'] / df['amount']
    volume_flow = volume_flow.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Volume Structure Divergence
    volume_structure_divergence = volume_concentration * volume_flow
    
    # Gap Behavior Transition Analysis
    # Gap Absorption Ratio
    gap_absorption = (df['close'] - df['open']) / (df['open'] - df['close'].shift(1))
    gap_absorption = gap_absorption.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Gap Persistence Signal
    gap_persistence = np.sign(df['open'] - df['close'].shift(1)) * np.sign(df['close'] - df['open'])
    
    # Gap Transition Quality
    gap_transition_quality = gap_absorption * gap_persistence
    
    # Range Expansion Momentum
    # Range Break Efficiency
    range_break_efficiency = (df['close'] - df['open']) / ((df['high'] - df['low']) * df['volume'] / df['amount'])
    range_break_efficiency = range_break_efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Range Expansion Signal
    range_expansion_signal = (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1)) - 1
    range_expansion_signal = range_expansion_signal.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Expansion Momentum Quality
    expansion_momentum_quality = range_break_efficiency * range_expansion_signal
    
    # Multi-Timeframe Volume Confirmation
    # Short-Term Volume Momentum (5-day)
    short_term_volume_momentum = df['volume'] / df['volume'].shift(5) - 1
    
    # Medium-Term Volume Trend (15-day)
    medium_term_volume_trend = df['volume'] / df['volume'].shift(15) - 1
    
    # Volume Timeframe Alignment
    volume_timeframe_alignment = np.sign(short_term_volume_momentum) * np.sign(medium_term_volume_trend)
    
    # Composite Transition Signal Construction
    # Combine Momentum Transition Components
    composite = momentum_transition * volume_structure_divergence
    
    # Apply Gap Transition Confirmation
    composite = composite * gap_transition_quality
    
    # Incorporate Range Expansion Momentum
    composite = composite * expansion_momentum_quality
    
    # Volume Persistence Structure
    # Analyze Volume Pattern Consistency
    volume_concentration_rolling_avg = volume_concentration.rolling(window=5, min_periods=1).mean()
    consistent_volume_days = volume_concentration.rolling(window=5, min_periods=1).apply(
        lambda x: (x > volume_concentration_rolling_avg.loc[x.index]).sum(), raw=False
    )
    
    # Apply Structure Weight
    final_signal = composite * (1 + consistent_volume_days / 8)
    
    # Apply Transition Quality Filters
    valid_signals = (
        (np.abs(accel_pattern) > 0.02) & 
        (volume_concentration > 1.5)
    )
    
    # Output Raw Transition Composite
    alpha_factor = final_signal.where(valid_signals, 0)
    
    return alpha_factor
