import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Momentum Fractal Framework alpha factor
    Combines momentum structure analysis with fractal pattern geometry
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(3, len(df)):
        current_data = df.iloc[:i+1]
        
        # Momentum Structure Analysis
        if i >= 1:
            directional_momentum = (current_data['close'].iloc[-1] - current_data['close'].iloc[-2]) / \
                                 (current_data['high'].iloc[-1] - current_data['low'].iloc[-1] + 1e-8)
        else:
            directional_momentum = 0
            
        if i >= 2:
            acceleration_momentum = (current_data['close'].iloc[-1] - 2*current_data['close'].iloc[-2] + current_data['close'].iloc[-3]) / \
                                  (current_data['high'].iloc[-2] - current_data['low'].iloc[-2] + 1e-8)
        else:
            acceleration_momentum = 0
            
        if i >= 3:
            curvature_momentum = (current_data['close'].iloc[-1] - 3*current_data['close'].iloc[-2] + 
                                3*current_data['close'].iloc[-3] - current_data['close'].iloc[-4]) / \
                               (current_data['high'].iloc[-3] - current_data['low'].iloc[-3] + 1e-8)
        else:
            curvature_momentum = 0
            
        # Fractal Pattern Geometry
        # Micro-Fractal: Price position relative to range
        micro_fractal = ((current_data['high'].iloc[-1] + current_data['low'].iloc[-1]) / 2 - current_data['close'].iloc[-1]) / \
                       (current_data['high'].iloc[-1] - current_data['low'].iloc[-1] + 1e-8)
        
        # Meso-Fractal: Triangle area pattern similarity (3-day pattern)
        if i >= 2:
            # Calculate triangle area using three consecutive points
            x1, y1 = 0, current_data['close'].iloc[-3]
            x2, y2 = 1, current_data['close'].iloc[-2]
            x3, y3 = 2, current_data['close'].iloc[-1]
            
            triangle_area = abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2)
            range_normalizer = max(current_data['high'].iloc[-3:-1].max() - current_data['low'].iloc[-3:-1].min(), 1e-8)
            meso_fractal = triangle_area / range_normalizer
        else:
            meso_fractal = 0
            
        # Macro-Fractal: Price curve curvature (5-day window)
        if i >= 4:
            window_data = current_data['close'].iloc[-5:]
            if len(window_data) == 5:
                # Fit quadratic curve to measure curvature
                x = np.arange(5)
                y = window_data.values
                coeffs = np.polyfit(x, y, 2)
                macro_fractal = coeffs[0] * 2  # Second derivative coefficient
            else:
                macro_fractal = 0
        else:
            macro_fractal = 0
            
        # Momentum-Pattern Integration
        # Convergent Geometry + Positive Momentum
        convergent_score = (abs(micro_fractal) + abs(meso_fractal) + abs(macro_fractal)) / 3
        momentum_strength = (directional_momentum + acceleration_momentum + curvature_momentum) / 3
        
        # Pattern stability measure (lower values = more stable patterns)
        pattern_stability = np.std([micro_fractal, meso_fractal, macro_fractal])
        
        # Final alpha calculation
        if convergent_score < 0.1 and momentum_strength > 0:  # Convergent + Positive Momentum
            alpha_value = momentum_strength * (1 - pattern_stability)
        elif convergent_score > 0.3 and momentum_strength < 0:  # Divergent + Negative Momentum
            alpha_value = momentum_strength * pattern_stability
        elif pattern_stability < 0.15 and acceleration_momentum > 0:  # Stable + Accelerating
            alpha_value = acceleration_momentum * (1 - convergent_score)
        else:
            # Neutral state - weighted combination
            alpha_value = (directional_momentum * 0.4 + 
                          acceleration_momentum * 0.3 + 
                          curvature_momentum * 0.3) * (1 - pattern_stability)
        
        alpha.iloc[i] = alpha_value
    
    # Fill initial values
    alpha = alpha.fillna(0)
    
    return alpha
