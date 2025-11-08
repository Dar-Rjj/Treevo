import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal Complexity-Adaptive Momentum Divergence factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    for i in range(5, len(data)):
        current_data = data.iloc[:i+1]  # Only use current and past data
        
        # 1. Fractal Momentum Complexity Analysis
        # Asymmetric Momentum Fractal Dynamics
        upside_path = 0
        downside_path = 0
        
        for j in range(5):
            if i-j-1 >= 0:
                price_diff = current_data['close'].iloc[i-j] - current_data['close'].iloc[i-j-1]
                upside_path += max(0, price_diff)
                downside_path += abs(min(0, price_diff))
        
        # Avoid division by zero
        if downside_path == 0:
            momentum_asymmetry_ratio = upside_path / 0.001
        else:
            momentum_asymmetry_ratio = upside_path / downside_path
        
        # Multi-Timeframe Complexity Divergence
        # Short-term complexity (3-day)
        short_term_price_changes = 0
        short_term_ranges = 0
        
        for j in range(3):
            if i-j-1 >= 0:
                short_term_price_changes += abs(current_data['close'].iloc[i-j] - current_data['close'].iloc[i-j-1])
                short_term_ranges += current_data['high'].iloc[i-j] - current_data['low'].iloc[i-j]
        
        if short_term_ranges > 0:
            short_term_complexity = 1 + np.log(short_term_price_changes + 1) / np.log(short_term_ranges + 1)
        else:
            short_term_complexity = 1.0
        
        # Medium-term complexity (5-day)
        medium_term_price_changes = 0
        medium_term_ranges = 0
        
        for j in range(5):
            if i-j-1 >= 0:
                medium_term_price_changes += abs(current_data['close'].iloc[i-j] - current_data['close'].iloc[i-j-1])
                medium_term_ranges += current_data['high'].iloc[i-j] - current_data['low'].iloc[i-j]
        
        if medium_term_ranges > 0:
            medium_term_complexity = 1 + np.log(medium_term_price_changes + 1) / np.log(medium_term_ranges + 1)
        else:
            medium_term_complexity = 1.0
        
        if medium_term_complexity > 0:
            complexity_divergence_ratio = short_term_complexity / medium_term_complexity
        else:
            complexity_divergence_ratio = 1.0
        
        # Momentum Quality Assessment
        momentum_consistency = 0
        for j in range(5):
            if i-j-1 >= 0 and current_data['close'].iloc[i-j] > current_data['close'].iloc[i-j-1]:
                momentum_consistency += 1
        momentum_consistency_score = momentum_consistency / 5
        
        total_momentum = upside_path + downside_path
        if total_momentum > 0:
            fractal_momentum_strength = (upside_path - downside_path) / total_momentum
        else:
            fractal_momentum_strength = 0
        
        if complexity_divergence_ratio > 0:
            complexity_momentum_alignment = momentum_asymmetry_ratio / complexity_divergence_ratio
        else:
            complexity_momentum_alignment = momentum_asymmetry_ratio
        
        # 2. Microstructure Volume Complexity Patterns
        # Volume-Price Efficiency Complexity
        daily_efficiency_ratio = abs(current_data['close'].iloc[i] - current_data['open'].iloc[i]) / (current_data['volume'].iloc[i] + 0.001)
        
        # Volume Complexity (3-day)
        volume_changes = 0
        for j in range(3):
            if i-j-1 >= 0:
                volume_changes += abs(current_data['volume'].iloc[i-j] - current_data['volume'].iloc[i-j-1])
        
        if volume_changes > 0:
            volume_complexity = 1 + np.log(volume_changes + 1) / np.log(volume_changes + 1)
        else:
            volume_complexity = 1.0
        
        if volume_complexity > 0:
            efficiency_complexity_divergence = daily_efficiency_ratio / volume_complexity
        else:
            efficiency_complexity_divergence = daily_efficiency_ratio
        
        # Volume Fractal Distribution Complexity
        # Simplified volume concentration (using daily data only)
        volume_concentration_complexity = current_data['volume'].iloc[i] / (current_data['volume'].iloc[max(0, i-4):i+1].sum() + 0.001)
        
        # Simplified volume momentum
        if i-5 >= 0:
            volume_momentum_complexity = current_data['volume'].iloc[i] / (current_data['volume'].iloc[i-5] + 0.001)
        else:
            volume_momentum_complexity = 1.0
        
        # 3. Volatility-Complexity Microstructure Interaction
        # Asymmetric Volatility Complexity
        daily_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
        daily_volatility_complexity = daily_range / (current_data['close'].iloc[i] + 0.001)
        
        # Volatility cluster detection (simplified)
        volatility_cluster_count = 0
        for j in range(min(5, i+1)):
            if i-j >= 0:
                range_pct = (current_data['high'].iloc[i-j] - current_data['low'].iloc[i-j]) / current_data['close'].iloc[i-j]
                if range_pct > 0.02:
                    volatility_cluster_count += 1
        
        # 4. Complexity-Adaptive Factor Construction
        # Base Fractal Complexity Component
        if complexity_divergence_ratio > 1.2:
            momentum_complexity_score = 1.8
        elif complexity_divergence_ratio < 0.8:
            momentum_complexity_score = 0.7
        else:
            momentum_complexity_score = 1.0
        
        # Apply momentum quality adjustments
        base_component = momentum_complexity_score * momentum_consistency_score * (1 + fractal_momentum_strength)
        
        # Microstructure Complexity Component
        if efficiency_complexity_divergence > 1.1:
            efficiency_pattern_score = 1.4
        elif efficiency_complexity_divergence < 0.9:
            efficiency_pattern_score = 0.8
        else:
            efficiency_pattern_score = 1.0
        
        microstructure_component = efficiency_pattern_score
        
        # Volume complexity adjustments
        if volume_concentration_complexity > 0.3:
            microstructure_component *= 1.3
        if volume_momentum_complexity > 1.1:
            microstructure_component *= 1.4
        
        # Volatility-Complexity Adaptation
        volatility_multiplier = 1.0
        if volatility_cluster_count > 3:
            volatility_multiplier *= 1.5
        if daily_volatility_complexity > np.percentile(current_data['high'].iloc[:i+1] - current_data['low'].iloc[:i+1], 70) / (current_data['close'].iloc[i] + 0.001):
            volatility_multiplier *= 1.3
        
        # Complexity-specific combinations
        if complexity_divergence_ratio > 1.2 and efficiency_complexity_divergence > 1.1:
            volatility_multiplier *= 2.0
        elif complexity_divergence_ratio < 0.8 and efficiency_complexity_divergence < 0.9:
            volatility_multiplier *= 0.5
        elif 0.8 <= complexity_divergence_ratio <= 1.2 and volatility_cluster_count > 3:
            volatility_multiplier *= 1.8
        
        # Final factor calculation
        factor_value = base_component * microstructure_component * volatility_multiplier
        
        # Apply complexity-momentum alignment adjustment
        factor.iloc[i] = factor_value * (1 + 0.1 * complexity_momentum_alignment)
    
    # Fill initial NaN values with neutral factor value
    factor = factor.fillna(1.0)
    
    return factor
