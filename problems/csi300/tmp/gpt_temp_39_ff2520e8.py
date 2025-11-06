import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 5:  # Need at least 5 days for calculations
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[i]
        past_data = df.iloc[max(0, i-4):i+1]  # Current and past 4 days
        
        # Multi-Scale Volatility-Fractal Classification
        short_term_vol = (current_data['high'] - current_data['low']) / current_data['close']
        medium_term_high = past_data['high'].max()
        medium_term_low = past_data['low'].min()
        medium_term_vol = (medium_term_high - medium_term_low) / current_data['close']
        
        fractal_vol_switch = short_term_vol / medium_term_vol if medium_term_vol != 0 else 0
        
        # Fractal Range Consistency
        daily_ranges = past_data['high'] - past_data['low']
        fractal_range_consistency = daily_ranges.std() / daily_ranges.mean() if daily_ranges.mean() != 0 else 0
        
        # Volatility Regime Classification
        if fractal_vol_switch > 1.5:
            volatility_regime = 'high'
        elif fractal_vol_switch < 0.67:
            volatility_regime = 'low'
        else:
            volatility_regime = 'transition'
        
        # Asymmetric Pressure-Volume Flow System
        high_low_range = current_data['high'] - current_data['low']
        if high_low_range != 0:
            upward_pressure = (current_data['close'] - current_data['low']) / high_low_range
            downward_pressure = (current_data['high'] - current_data['close']) / high_low_range
            net_pressure_flow = upward_pressure - downward_pressure
        else:
            net_pressure_flow = 0
        
        # Flow Momentum (3-day change in pressure flow)
        if i >= 3:
            past_pressure = (df.iloc[i-3]['close'] - df.iloc[i-3]['low']) / (df.iloc[i-3]['high'] - df.iloc[i-3]['low']) if (df.iloc[i-3]['high'] - df.iloc[i-3]['low']) != 0 else 0
            flow_momentum = net_pressure_flow - past_pressure
        else:
            flow_momentum = 0
        
        # Volume Acceleration
        if i >= 1 and df.iloc[i-1]['volume'] != 0:
            volume_acceleration = (current_data['volume'] / df.iloc[i-1]['volume']) * np.sign(current_data['close'] - current_data['open'])
        else:
            volume_acceleration = 0
        
        pressure_volume_correlation = net_pressure_flow * volume_acceleration
        
        # Cumulative Pressure Flow (4-day sum)
        cumulative_pressure = 0
        for j in range(max(0, i-4), i+1):
            if j < len(df):
                high_low_j = df.iloc[j]['high'] - df.iloc[j]['low']
                if high_low_j != 0:
                    pressure_j = (df.iloc[j]['close'] - df.iloc[j]['low']) / high_low_j
                    cumulative_pressure += pressure_j
        
        volume_weighted_flow = cumulative_pressure * current_data['volume']
        
        # Fractal Gap Momentum with Range Efficiency
        if i >= 1:
            overnight_gap = (current_data['open'] - df.iloc[i-1]['close']) / df.iloc[i-1]['close'] if df.iloc[i-1]['close'] != 0 else 0
        else:
            overnight_gap = 0
        
        intraday_efficiency = abs(current_data['close'] - current_data['open']) / high_low_range if high_low_range != 0 else 0
        fractal_gap_momentum = overnight_gap * intraday_efficiency
        
        # Volume-Anchored Fractal Gaps
        if i >= 1 and df.iloc[i-1]['volume'] != 0:
            gap_volume_ratio = current_data['volume'] / df.iloc[i-1]['volume']
        else:
            gap_volume_ratio = 1
        
        volume_weighted_gap = fractal_gap_momentum * gap_volume_ratio
        
        # Fractal Range Expansion with Pressure Confirmation
        if i >= 1:
            prev_range = df.iloc[i-1]['high'] - df.iloc[i-1]['low']
            if prev_range != 0:
                range_expansion = high_low_range / prev_range
            else:
                range_expansion = 1
        else:
            range_expansion = 1
        
        range_efficiency = abs(current_data['close'] - current_data['open']) / high_low_range if high_low_range != 0 else 0
        expansion_quality = range_expansion * range_efficiency
        
        # Pressure-Confirmed Expansion
        expansion_pressure_alignment = expansion_quality * net_pressure_flow
        expansion_volume_sync = range_expansion * current_data['volume']
        
        # Adaptive Fractal Momentum Synthesis
        if volatility_regime == 'high':
            # Emphasize pressure flow and fractal gap momentum
            momentum_component = 0.4 * pressure_volume_correlation + 0.3 * volume_weighted_gap + 0.2 * expansion_pressure_alignment + 0.1 * fractal_range_consistency
        elif volatility_regime == 'low':
            # Focus on fractal range expansion efficiency
            momentum_component = 0.5 * expansion_quality + 0.3 * range_efficiency + 0.2 * volume_weighted_flow
        else:  # transition regime
            # Balanced weighting across fractal components
            momentum_component = 0.25 * pressure_volume_correlation + 0.25 * volume_weighted_gap + 0.25 * expansion_pressure_alignment + 0.25 * expansion_quality
        
        # Final alpha factor with confidence adjustment
        # High confidence signals: strong alignment across multiple components
        signal_strength = (abs(pressure_volume_correlation) + abs(volume_weighted_gap) + abs(expansion_pressure_alignment)) / 3
        
        if signal_strength > 0.1:  # Strong signal threshold
            confidence_multiplier = 1.5
        else:
            confidence_multiplier = 0.5
        
        result.iloc[i] = momentum_component * confidence_multiplier
    
    return result
