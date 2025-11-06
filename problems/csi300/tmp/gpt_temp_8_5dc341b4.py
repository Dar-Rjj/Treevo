import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Asset Fractal Interaction Framework alpha factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate all intermediate components
    for i in range(11, len(df)):
        if i < 11:
            continue
            
        # Extract current and historical data
        current = df.iloc[i]
        prev_1 = df.iloc[i-1] if i >= 1 else None
        prev_2 = df.iloc[i-2] if i >= 2 else None
        prev_3 = df.iloc[i-3] if i >= 3 else None
        prev_4 = df.iloc[i-4] if i >= 4 else None
        prev_5 = df.iloc[i-5] if i >= 5 else None
        prev_8 = df.iloc[i-8] if i >= 8 else None
        prev_10 = df.iloc[i-10] if i >= 10 else None
        prev_11 = df.iloc[i-11] if i >= 11 else None
        
        # Skip if any required data is missing
        if any(x is None for x in [prev_1, prev_2, prev_3, prev_4, prev_5, prev_8, prev_10, prev_11]):
            continue
        
        # Relative Fractal Momentum Dynamics
        cross_timeframe_momentum_ratio = (
            (current['close']/prev_3['close'] - current['close']/prev_10['close']) / 
            (prev_1['close']/prev_4['close'] - prev_1['close']/prev_11['close'])
        )
        
        fractal_momentum_persistence = (
            np.sign(current['close']/prev_3['close'] - current['close']/prev_10['close']) * 
            abs(current['close']/prev_3['close'] - current['close']/prev_10['close']) * 
            current['volume']/prev_1['volume']
        )
        
        volatility_adjusted_fractal_acceleration = (
            (current['close']/prev_5['close'] - current['close']/prev_10['close']) * 
            (prev_1['high'] - prev_1['low'])/(prev_3['high'] - prev_3['low'])
        )
        
        # Volume-Fractal Regime Detection
        fractal_volume_regime_shift = (
            (current['volume']/prev_3['volume'] - current['volume']/prev_8['volume']) * 
            (prev_1['high'] - prev_1['low'])/(prev_4['high'] - prev_4['low'])
        )
        
        trade_size_fractal_efficiency = (
            (current['amount']/current['volume']) * 
            abs(current['close'] - prev_1['close'])/(current['high'] - current['low']) * 
            current['volume']/prev_2['volume']
        )
        
        fractal_volume_compression = (
            (current['volume']/(current['high'] - current['low'])) / 
            (prev_2['volume']/(prev_2['high'] - prev_2['low']))
        )
        
        # Asymmetric Fractal Microstructure Patterns
        fractal_bid_ask_pressure = (
            (current['high'] - max(current['open'], current['close'])) - 
            (min(current['open'], current['close']) - current['low']) * 
            current['volume']/prev_1['volume']
        )
        
        opening_gap_fractal_absorption = (
            (current['open'] - prev_1['close'])/(prev_1['high'] - prev_1['low']) * 
            current['volume']/prev_2['volume']
        )
        
        closing_fractal_pressure = (
            (current['close'] - current['open'])/(current['high'] - current['low']) * 
            current['volume']/prev_1['volume']
        )
        
        # Fractal Breakout Confirmation System
        volume_confirmed_fractal_breakout = (
            current['volume']/prev_2['volume'] * 
            (prev_1['high'] - prev_1['low'])/(prev_4['high'] - prev_4['low'])
            if current['high'] > prev_1['high'] and current['close'] > prev_1['close'] else 0
        )
        
        fractal_gap_momentum_persistence = (
            (current['open']/prev_1['close'] - 1) * 
            current['volume']/prev_1['volume'] * 
            np.sign(current['open']/prev_1['close'] - 1)
        )
        
        range_expansion_fractal_flow = (
            (current['high']/current['low'] - prev_1['high']/prev_1['low']) * 
            current['volume']/prev_1['volume']
        )
        
        # Fractal Flow Efficiency Metrics
        intraday_fractal_efficiency = (
            abs(current['close'] - current['open'])/(current['high'] - current['low']) * 
            current['volume']/prev_2['volume']
        )
        
        fractal_efficiency_momentum = (
            (intraday_fractal_efficiency / (
                abs(prev_2['close'] - prev_2['open'])/(prev_2['high'] - prev_2['low']) * 
                prev_2['volume']/prev_4['volume']
            ) - 1) * current['volume']/prev_1['volume']
        )
        
        volatility_fractal_compression = (
            (current['high'] - current['low'])/(prev_2['high'] - prev_2['low']) * 
            current['volume']/prev_1['volume']
        )
        
        # Core Fractal Interaction Components
        momentum_volume_fractal_synergy = fractal_momentum_persistence * fractal_volume_compression
        trade_size_fractal_alignment = trade_size_fractal_efficiency * fractal_efficiency_momentum
        microstructure_fractal_pressure = fractal_bid_ask_pressure * closing_fractal_pressure
        
        # Fractal Regime Classification
        high_efficiency_fractal_flow = intraday_fractal_efficiency * fractal_volume_regime_shift
        volume_driven_fractal_momentum = fractal_volume_compression * cross_timeframe_momentum_ratio
        range_breakout_fractal_confirmation = range_expansion_fractal_flow * volume_confirmed_fractal_breakout
        
        # Fractal Signal Validation
        multi_timeframe_fractal_alignment = momentum_volume_fractal_synergy * fractal_efficiency_momentum
        volume_microstructure_fractal_convergence = trade_size_fractal_alignment * microstructure_fractal_pressure
        breakout_efficiency_fractal_validation = volume_driven_fractal_momentum * high_efficiency_fractal_flow
        
        # Final Fractal Alpha Architecture
        primary_alpha = multi_timeframe_fractal_alignment * volume_microstructure_fractal_convergence
        secondary_alpha = breakout_efficiency_fractal_validation * range_breakout_fractal_confirmation
        
        # Composite Fractal Alpha with dynamic weighting
        regime_strength = abs(fractal_volume_regime_shift) + abs(intraday_fractal_efficiency)
        weight_primary = abs(fractal_volume_regime_shift) / regime_strength if regime_strength != 0 else 0.5
        weight_secondary = 1 - weight_primary
        
        composite_fractal_alpha = weight_primary * primary_alpha + weight_secondary * secondary_alpha
        
        alpha.iloc[i] = composite_fractal_alpha
    
    # Fill NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
