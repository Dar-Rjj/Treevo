import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(2, len(df)):
        if i < 4:  # Need at least 4 days for some calculations
            continue
            
        # Current day data
        open_t = df.iloc[i]['open']
        high_t = df.iloc[i]['high']
        low_t = df.iloc[i]['low']
        close_t = df.iloc[i]['close']
        volume_t = df.iloc[i]['volume']
        amount_t = df.iloc[i]['amount']
        
        # Previous day data
        close_t1 = df.iloc[i-1]['close']
        high_t1 = df.iloc[i-1]['high']
        low_t1 = df.iloc[i-1]['low']
        open_t1 = df.iloc[i-1]['open']
        volume_t1 = df.iloc[i-1]['volume']
        amount_t1 = df.iloc[i-1]['amount']
        
        # 3-day ago data for medium-term calculations
        close_t3 = df.iloc[i-3]['close']
        high_t3 = df.iloc[i-3]['high']
        low_t3 = df.iloc[i-3]['low']
        open_t3 = df.iloc[i-3]['open']
        volume_t3 = df.iloc[i-3]['volume']
        amount_t3 = df.iloc[i-3]['amount']
        
        # Avoid division by zero
        high_low_range = max(high_t - low_t, 1e-8)
        high_low_range_t1 = max(high_t1 - low_t1, 1e-8)
        high_low_range_t3 = max(high_t3 - low_t3, 1e-8)
        
        # Calculate max denominator for various components
        max_denom = max(high_low_range, abs(high_t - open_t), abs(low_t - open_t), 1e-8)
        max_denom_t1 = max(high_low_range_t1, abs(high_t1 - open_t1), abs(low_t1 - open_t1), 1e-8)
        max_denom_t3 = max(high_low_range_t3, abs(high_t3 - open_t3), abs(low_t3 - open_t3), 1e-8)
        
        # Bidirectional Flow Asymmetry Components
        # Aggressive Buying Pressure
        high_side_intensity = ((high_t - open_t) / high_low_range) * volume_t * ((close_t - open_t) / max(open_t, 1e-8))
        
        closing_momentum = ((close_t - open_t) / high_low_range) * (volume_t / max(amount_t, 1e-8)) * ((high_t - close_t) / high_low_range)
        
        upper_fractal = ((high_t - close_t1) / max(close_t1, 1e-8)) * (volume_t / max_denom)
        
        aggressive_buying = high_side_intensity + closing_momentum + upper_fractal
        
        # Defensive Selling Absorption
        low_side_resilience = ((open_t - low_t) / high_low_range) * volume_t * ((open_t - close_t) / max(open_t, 1e-8))
        
        opening_support = ((open_t - low_t) / high_low_range) * (amount_t / max(volume_t, 1e-8)) * ((close_t - low_t) / high_low_range)
        
        lower_fractal = ((close_t1 - low_t) / max(close_t1, 1e-8)) * (volume_t / max_denom)
        
        defensive_selling = low_side_resilience + opening_support + lower_fractal
        
        # Intraday Flow Imbalance
        directional_asymmetry = ((high_t - open_t) - (open_t - low_t)) / high_low_range * volume_t * ((close_t - open_t) / max(open_t, 1e-8))
        
        flow_momentum_div = ((close_t - open_t) / high_low_range) * (amount_t / max(volume_t, 1e-8)) * ((high_t - open_t) - (open_t - low_t)) / high_low_range
        
        microstructure_flow = abs(close_t - open_t) / high_low_range * (volume_t / max(amount_t, 1e-8)) * (volume_t / max_denom)
        
        intraday_imbalance = directional_asymmetry + flow_momentum_div + microstructure_flow
        
        # Weighted flow composite
        weighted_flow = 0.4 * aggressive_buying + 0.35 * defensive_selling + 0.25 * intraday_imbalance
        
        # Fractal Regime Detection
        fractal_range = high_low_range / max(close_t1, 1e-8)
        
        # High-Fractal Regime Signals
        wide_range_flow = fractal_range * ((high_t - open_t) - (open_t - low_t)) / high_low_range * (volume_t / max(amount_t, 1e-8))
        
        fractal_adj_momentum = ((close_t - open_t) / high_low_range) * (volume_t / max_denom) * fractal_range
        
        extreme_flow = fractal_range * (volume_t / max(amount_t, 1e-8)) * ((close_t - open_t) / high_low_range) * ((high_t - open_t) - (open_t - low_t)) / high_low_range
        
        high_fractal = wide_range_flow + fractal_adj_momentum + extreme_flow
        
        # Low-Fractal Regime Signals
        tight_range_persistence = fractal_range * (volume_t / max_denom) * ((close_t - open_t) / max(open_t, 1e-8))
        
        micro_flow_accum = ((close_t - open_t) / max(open_t, 1e-8)) * (amount_t / max(volume_t, 1e-8)) * (1 - fractal_range) * (volume_t / max(volume_t1, 1e-8))
        
        silent_flow = (volume_t / max(volume_t1, 1e-8)) * ((close_t - open_t) / high_low_range) * (amount_t / max(volume_t, 1e-8)) * (1 - fractal_range)
        
        low_fractal = tight_range_persistence + micro_flow_accum + silent_flow
        
        # Fractal Transition Detection
        flow_regime_shift = ((high_t - open_t) - (open_t - low_t)) / high_low_range * (volume_t / max(volume_t1, 1e-8)) * ((close_t - open_t) / max(open_t, 1e-8))
        
        # Multi-Scale Flow Alignment
        # Short-term (2-day)
        daily_flow_momentum = ((close_t - open_t) / high_low_range - (close_t1 - open_t1) / high_low_range_t1) * (volume_t / max(volume_t1, 1e-8))
        
        volume_flow_align = (volume_t / max_denom - volume_t1 / max_denom_t1) * (amount_t / max(amount_t1, 1e-8))
        
        amount_flow_persistence = (amount_t / max(volume_t, 1e-8)) * ((close_t - open_t) / high_low_range) - (amount_t1 / max(volume_t1, 1e-8)) * ((close_t1 - open_t1) / high_low_range_t1)
        
        # Medium-term (4-day)
        multi_day_structure = ((close_t - open_t) / high_low_range - (close_t3 - open_t3) / high_low_range_t3) * (volume_t / max(volume_t3, 1e-8))
        
        # Flow Quality Assessment
        price_impact_flow = ((close_t - open_t) / high_low_range) * (volume_t / max(amount_t, 1e-8)) * ((high_t - open_t) - (open_t - low_t)) / high_low_range
        
        volume_quality = (volume_t / max_denom) * (amount_t / max(volume_t, 1e-8)) * ((close_t - open_t) / max(open_t, 1e-8))
        
        microstructure_ratio = ((close_t - open_t) / high_low_range) * (volume_t / max_denom) * (amount_t / max(volume_t, 1e-8))
        
        # Composite factors with quality enhancement
        fractal_filtered = weighted_flow * (1 - fractal_range)
        volume_confirmed = weighted_flow * (volume_t / max(volume_t1, 1e-8))
        
        microstructure_quality = weighted_flow * (volume_t / max_denom) * (amount_t / max(volume_t, 1e-8)) * ((high_t - open_t) - (open_t - low_t)) / high_low_range
        
        # Final composite alpha
        alpha_value = (
            0.3 * weighted_flow +
            0.2 * (high_fractal - low_fractal) +
            0.15 * (daily_flow_momentum + volume_flow_align) +
            0.15 * multi_day_structure +
            0.1 * (price_impact_flow + volume_quality) +
            0.05 * microstructure_ratio +
            0.05 * (fractal_filtered + volume_confirmed + microstructure_quality)
        )
        
        result.iloc[i] = alpha_value
    
    # Fill NaN values with 0
    result = result.fillna(0)
    
    return result
