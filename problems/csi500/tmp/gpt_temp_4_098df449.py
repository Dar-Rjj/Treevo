import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Fractal Momentum with Microstructure Anchoring alpha factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required minimum data length
    min_periods = 60
    
    for i in range(min_periods, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # 1. Multi-Scale Fractal Dimension Estimation
        # Short-term (1-5 days)
        short_term_range = []
        for j in range(max(0, i-4), i+1):
            if j >= 1:
                price_range = (current_data['high'].iloc[j] - current_data['low'].iloc[j]) / current_data['close'].iloc[j-1]
                short_term_range.append(price_range)
        
        short_term_complexity = np.std(short_term_range) if len(short_term_range) > 1 else 0
        
        # Medium-term (5-20 days)
        medium_term_fractal = []
        for j in range(max(0, i-19), i+1):
            if j >= 5:
                window = current_data.iloc[j-5:j+1]
                high_low_ratio = (window['high'].max() - window['low'].min()) / window['close'].iloc[-1]
                medium_term_fractal.append(high_low_ratio)
        
        medium_fractal_dim = np.mean(medium_term_fractal) if medium_term_fractal else 0
        
        # Long-term (20-60 days)
        if i >= 59:
            long_window = current_data.iloc[i-59:i+1]
            price_changes = long_window['close'].pct_change().dropna()
            hurst_exp = 0.5 + 0.5 * np.log(np.var(price_changes.iloc[30:]) / np.var(price_changes.iloc[:30])) / np.log(2) if len(price_changes) >= 30 else 0.5
            long_fractal_consistency = abs(hurst_exp - 0.5)
        else:
            long_fractal_consistency = 0
        
        # 2. Fractal Momentum Divergence
        # Multi-scale momentum alignment
        short_momentum = current_data['close'].iloc[i] / current_data['close'].iloc[max(0, i-4)] - 1 if i >= 4 else 0
        medium_momentum = current_data['close'].iloc[i] / current_data['close'].iloc[max(0, i-19)] - 1 if i >= 19 else 0
        
        momentum_alignment = np.sign(short_momentum) == np.sign(medium_momentum)
        momentum_strength = (abs(short_momentum) + abs(medium_momentum)) / 2 if momentum_alignment else 0
        
        # 3. Volume-Fractal Interaction
        # Recent volume clustering
        recent_volume = current_data['volume'].iloc[max(0, i-9):i+1]
        volume_clustering = np.std(recent_volume) / np.mean(recent_volume) if len(recent_volume) > 1 else 0
        
        # Volume confirmation of momentum
        volume_momentum_corr = 0
        if i >= 9:
            price_changes = current_data['close'].iloc[i-9:i+1].pct_change().dropna()
            volume_changes = current_data['volume'].iloc[i-9:i+1].pct_change().dropna()
            if len(price_changes) == len(volume_changes) and len(price_changes) > 1:
                volume_momentum_corr = np.corrcoef(price_changes, volume_changes)[0,1] if not np.isnan(np.corrcoef(price_changes, volume_changes)[0,1]) else 0
        
        # 4. Microstructure Anchoring Mechanisms
        # Opening auction effects
        if i >= 1:
            open_gap = (current_data['open'].iloc[i] - current_data['close'].iloc[i-1]) / current_data['close'].iloc[i-1]
            gap_significance = abs(open_gap) / (current_data['high'].iloc[i-1] - current_data['low'].iloc[i-1]) * current_data['close'].iloc[i-1] if (current_data['high'].iloc[i-1] - current_data['low'].iloc[i-1]) > 0 else 0
        else:
            gap_significance = 0
        
        # Intraday microstructure anchors
        recent_highs = current_data['high'].iloc[max(0, i-4):i+1]
        recent_lows = current_data['low'].iloc[max(0, i-4):i+1]
        
        resistance_level = np.mean(recent_highs)
        support_level = np.mean(recent_lows)
        
        price_vs_resistance = (current_data['close'].iloc[i] - resistance_level) / resistance_level
        price_vs_support = (current_data['close'].iloc[i] - support_level) / support_level
        
        # 5. Fractal-Microstructure Convergence
        # Multi-scale anchor alignment
        if i >= 19:
            medium_high = current_data['high'].iloc[i-19:i+1].max()
            medium_low = current_data['low'].iloc[i-19:i+1].min()
            
            fractal_resistance = (medium_high + resistance_level) / 2
            fractal_support = (medium_low + support_level) / 2
            
            fractal_micro_alignment = 1 - (abs(fractal_resistance - resistance_level) + abs(fractal_support - support_level)) / (current_data['close'].iloc[i] * 2) if current_data['close'].iloc[i] > 0 else 0
        else:
            fractal_micro_alignment = 0
        
        # Volume concentration at key levels
        recent_close = current_data['close'].iloc[i]
        level_proximity = min(abs(recent_close - resistance_level), abs(recent_close - support_level)) / recent_close if recent_close > 0 else 1
        
        volume_at_levels = 0
        if level_proximity < 0.02:  # Within 2% of key level
            volume_at_levels = current_data['volume'].iloc[i] / np.mean(current_data['volume'].iloc[max(0, i-4):i+1]) if np.mean(current_data['volume'].iloc[max(0, i-4):i+1]) > 0 else 1
        
        # 6. Integrated Alpha Construction
        # Fractal Momentum Strength Score
        fractal_consistency = (1 - short_term_complexity) * 0.3 + medium_fractal_dim * 0.4 + long_fractal_consistency * 0.3
        
        # Microstructure-Validated Momentum
        microstructure_momentum = momentum_strength * (1 + volume_momentum_corr) * (1 - gap_significance)
        
        # Anchor Quality Assessment
        anchor_strength = volume_at_levels * (1 - level_proximity * 10)  # Higher volume closer to levels
        
        # Final Alpha Factor
        fractal_micro_convergence = fractal_micro_alignment * fractal_consistency
        
        alpha_value = (
            fractal_micro_convergence * 0.4 +
            microstructure_momentum * 0.3 +
            anchor_strength * 0.2 +
            volume_clustering * 0.1
        )
        
        result.iloc[i] = alpha_value
    
    # Fill initial values with 0
    result = result.fillna(0)
    
    return result
