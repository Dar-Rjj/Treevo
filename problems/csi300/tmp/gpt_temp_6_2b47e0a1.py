import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate turnover (volume * close)
    data['turnover'] = data['volume'] * data['close']
    
    # Calculate true range
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    
    for i in range(len(data)):
        if i < 20:  # Need sufficient history
            result.iloc[i] = 0
            continue
            
        current_data = data.iloc[:i+1]
        
        # Multi-Scale Fractal Turnover Analysis
        # Price fractals
        micro_price_fractal = (current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) / \
                             (current_data['high'].iloc[-2] - current_data['low'].iloc[-2]) if current_data['high'].iloc[-2] != current_data['low'].iloc[-2] else 1.0
        
        meso_high_current = current_data['high'].iloc[-3:].max()
        meso_low_current = current_data['low'].iloc[-3:].min()
        meso_high_prev = current_data['high'].iloc[-6:-3].max()
        meso_low_prev = current_data['low'].iloc[-6:-3].min()
        meso_price_fractal = (meso_high_current - meso_low_current) / (meso_high_prev - meso_low_prev) if meso_high_prev != meso_low_prev else 1.0
        
        macro_high_current = current_data['high'].iloc[-5:].max()
        macro_low_current = current_data['low'].iloc[-5:].min()
        macro_high_prev = current_data['high'].iloc[-10:-5].max()
        macro_low_prev = current_data['low'].iloc[-10:-5].min()
        macro_price_fractal = (macro_high_current - macro_low_current) / (macro_high_prev - macro_low_prev) if macro_high_prev != macro_low_prev else 1.0
        
        # Turnover fractals
        short_turnover_fractal = current_data['turnover'].iloc[-1] / current_data['turnover'].iloc[-3:-1].mean() if current_data['turnover'].iloc[-3:-1].mean() > 0 else 1.0
        medium_turnover_fractal = current_data['turnover'].iloc[-3:].mean() / current_data['turnover'].iloc[-6:-3].mean() if current_data['turnover'].iloc[-6:-3].mean() > 0 else 1.0
        long_turnover_fractal = current_data['turnover'].iloc[-5:].mean() / current_data['turnover'].iloc[-10:-5].mean() if current_data['turnover'].iloc[-10:-5].mean() > 0 else 1.0
        
        # Fractal alignment
        micro_alignment = micro_price_fractal / short_turnover_fractal if short_turnover_fractal != 0 else 1.0
        meso_alignment = meso_price_fractal / medium_turnover_fractal if medium_turnover_fractal != 0 else 1.0
        macro_alignment = macro_price_fractal / long_turnover_fractal if long_turnover_fractal != 0 else 1.0
        
        # Momentum-Turnover Divergence Detection
        # Multi-scale momentum
        short_momentum_range = current_data['high'].iloc[-3:].max() - current_data['low'].iloc[-3:].min()
        short_momentum = (current_data['close'].iloc[-1] - current_data['close'].iloc[-3]) / short_momentum_range if short_momentum_range > 0 else 0
        
        medium_momentum_range = current_data['high'].iloc[-5:].max() - current_data['low'].iloc[-5:].min()
        medium_momentum = (current_data['close'].iloc[-1] - current_data['close'].iloc[-5]) / medium_momentum_range if medium_momentum_range > 0 else 0
        
        long_momentum_range = current_data['high'].iloc[-10:].max() - current_data['low'].iloc[-10:].min()
        long_momentum = (current_data['close'].iloc[-1] - current_data['close'].iloc[-10]) / long_momentum_range if long_momentum_range > 0 else 0
        
        # Momentum divergence
        short_medium_divergence = short_momentum - medium_momentum
        medium_long_divergence = medium_momentum - long_momentum
        cross_scale_divergence = np.sign(short_momentum * medium_momentum * long_momentum) * (abs(short_momentum * medium_momentum * long_momentum)) ** (1/3) if short_momentum * medium_momentum * long_momentum != 0 else 0
        
        # Turnover confirmation
        turnover_acceleration = medium_turnover_fractal / long_turnover_fractal if long_turnover_fractal != 0 else 1.0
        momentum_turnover_alignment = cross_scale_divergence * turnover_acceleration
        divergence_strength = abs(short_medium_divergence) + abs(medium_long_divergence)
        
        # Volatility-Weighted Boundary Analysis
        upper_boundary = (current_data['high'].iloc[-1] - current_data['high'].iloc[-6:-1].max()) / (current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) if (current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) > 0 else 0
        lower_boundary = (current_data['low'].iloc[-6:-1].min() - current_data['low'].iloc[-1]) / (current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) if (current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) > 0 else 0
        boundary_asymmetry = upper_boundary / lower_boundary if lower_boundary != 0 else 1.0
        
        # Volatility components
        short_term_volatility = current_data['true_range'].iloc[-5:].mean()
        long_term_volatility = current_data['true_range'].iloc[-20:].mean()
        
        # Volatility-boundary interaction
        upper_boundary_volatility = upper_boundary * short_term_volatility
        lower_boundary_volatility = lower_boundary * short_term_volatility
        volatility_adjusted_boundaries = (upper_boundary_volatility - lower_boundary_volatility) / long_term_volatility if long_term_volatility > 0 else 0
        
        # Intraday Efficiency with Turnover Confirmation
        opening_efficiency = (current_data['open'].iloc[-1] - current_data['close'].iloc[-2]) / (current_data['high'].iloc[-2] - current_data['low'].iloc[-2]) if (current_data['high'].iloc[-2] - current_data['low'].iloc[-2]) > 0 else 0
        closing_efficiency = (current_data['close'].iloc[-1] - current_data['open'].iloc[-1]) / (current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) if (current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) > 0 else 0
        efficiency_ratio = closing_efficiency / abs(opening_efficiency) if opening_efficiency != 0 else 1.0
        
        morning_turnover_intensity = current_data['volume'].iloc[-2] / current_data['volume'].iloc[-4:-2].mean() if current_data['volume'].iloc[-4:-2].mean() > 0 else 1.0
        afternoon_turnover_intensity = current_data['volume'].iloc[-1] / current_data['volume'].iloc[-2] if current_data['volume'].iloc[-2] > 0 else 1.0
        turnover_efficiency = afternoon_turnover_intensity * efficiency_ratio
        
        # Efficiency-turnover alignment detection
        positive_alignment = efficiency_ratio > 1 and turnover_efficiency > 1
        negative_alignment = efficiency_ratio < 1 and turnover_efficiency < 1
        mixed_alignment = not (positive_alignment or negative_alignment)
        
        # Fractal Regime Classification
        price_expansion = all([micro_price_fractal > 1.1, meso_price_fractal > 1.1, macro_price_fractal > 1.1])
        turnover_expansion = all([short_turnover_fractal > 1.1, medium_turnover_fractal > 1.1, long_turnover_fractal > 1.1])
        synchronized_expansion = price_expansion and turnover_expansion
        
        price_contraction = all([micro_price_fractal < 0.9, meso_price_fractal < 0.9, macro_price_fractal < 0.9])
        turnover_contraction = all([short_turnover_fractal < 0.9, medium_turnover_fractal < 0.9, long_turnover_fractal < 0.9])
        synchronized_contraction = price_contraction and turnover_contraction
        
        # Adaptive Alpha Synthesis
        # Regime-based divergence weighting
        if synchronized_expansion:
            divergence_weight = 0.6 * medium_long_divergence + 0.4 * short_medium_divergence
        elif synchronized_contraction:
            divergence_weight = 0.7 * short_medium_divergence + 0.3 * medium_long_divergence
        else:
            divergence_weight = (short_medium_divergence + medium_long_divergence + cross_scale_divergence) / 3
        
        # Volatility-boundary adjustments
        if short_term_volatility > long_term_volatility:
            divergence_strength *= 0.8
        elif short_term_volatility < long_term_volatility:
            divergence_strength *= 1.2
        
        # Efficiency-turnover overlays
        if positive_alignment:
            momentum_bias = 1.2
        elif negative_alignment:
            momentum_bias = 0.8
        else:
            momentum_bias = 1.0
        
        # Fractal alignment finalization
        alignments = [micro_alignment, meso_alignment, macro_alignment]
        strong_alignment = all(a > 0.9 for a in alignments)
        moderate_alignment = all(0.7 <= a <= 0.9 for a in alignments)
        
        if strong_alignment:
            composite = (divergence_weight * divergence_strength * momentum_bias * 1.3 + 
                        momentum_turnover_alignment + volatility_adjusted_boundaries) / 3
        elif moderate_alignment:
            top_two = sorted(alignments, reverse=True)[:2]
            composite = (divergence_weight * divergence_strength * momentum_bias + 
                        np.mean(top_two) * momentum_turnover_alignment) / 2
        else:
            composite = divergence_weight * divergence_strength * momentum_bias
        
        # Dynamic smoothing based on regime stability
        regime_stable = synchronized_expansion or synchronized_contraction
        regime_transition = (price_expansion and turnover_contraction) or (price_contraction and turnover_expansion)
        
        if regime_stable and i >= 3:
            # Apply 3-day moving average for stable regimes
            if i == 3:
                result.iloc[i] = composite
            else:
                result.iloc[i] = (composite + result.iloc[i-1] + result.iloc[i-2]) / 3
        elif regime_transition:
            # Use raw values with 1-day lag for transitions
            result.iloc[i] = composite
        else:
            # Apply minimum of current and previous for volatile regimes
            if i == 0:
                result.iloc[i] = composite
            else:
                result.iloc[i] = min(composite, result.iloc[i-1])
    
    return result
