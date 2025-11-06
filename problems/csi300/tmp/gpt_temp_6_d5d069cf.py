import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Calculate required components with proper shifting to avoid lookahead
    for i in range(3, len(df)):
        current_data = df.iloc[i]
        prev1_data = df.iloc[i-1]
        prev2_data = df.iloc[i-2]
        prev3_data = df.iloc[i-3]
        
        # Asymmetric Momentum Framework
        # Directional Momentum Bias
        high_low_range = current_data['high'] - current_data['low']
        if high_low_range > 0:
            upward_momentum = (current_data['close'] - current_data['low']) / high_low_range
            downward_resistance = (current_data['high'] - current_data['close']) / high_low_range
            momentum_asymmetry = upward_momentum - downward_resistance
        else:
            momentum_asymmetry = 0
        
        # Multi-timeframe Momentum
        prev1_range = prev1_data['high'] - prev1_data['low']
        prev2_range = prev2_data['high'] - prev2_data['low']
        prev3_range = prev3_data['high'] - prev3_data['low']
        
        if prev1_range > 0:
            short_term_momentum = (current_data['close'] - prev1_data['close']) / prev1_range
        else:
            short_term_momentum = 0
            
        avg_prev_ranges = (prev1_range + prev2_range + prev3_range) / 3
        if avg_prev_ranges > 0:
            medium_term_momentum = (current_data['close'] - prev3_data['close']) / avg_prev_ranges
        else:
            medium_term_momentum = 0
            
        momentum_convergence = short_term_momentum * medium_term_momentum
        
        # Volume-Weighted Momentum
        if prev1_data['volume'] > 0:
            volume_momentum = (current_data['volume'] / prev1_data['volume']) * short_term_momentum
        else:
            volume_momentum = 0
            
        if prev1_data['amount'] > 0:
            amount_momentum = (current_data['amount'] / prev1_data['amount']) * short_term_momentum
        else:
            amount_momentum = 0
        
        # Liquidity Fracture Dynamics
        # Volume Fracture
        avg_volume = (current_data['volume'] + prev1_data['volume']) / 2
        if avg_volume > 0:
            volume_gap = abs(current_data['volume'] - prev1_data['volume']) / avg_volume
        else:
            volume_gap = 0
            
        avg_volume_3 = (prev1_data['volume'] + prev2_data['volume'] + prev3_data['volume']) / 3
        if avg_volume_3 > 0:
            volume_cluster = current_data['volume'] / avg_volume_3
        else:
            volume_cluster = 0
        
        # Amount Fracture
        avg_amount = (current_data['amount'] + prev1_data['amount']) / 2
        if avg_amount > 0:
            amount_break = abs(current_data['amount'] - prev1_data['amount']) / avg_amount
        else:
            amount_break = 0
            
        if prev1_data['amount'] > 0 and prev1_data['volume'] > 0:
            amount_divergence = (current_data['amount'] / prev1_data['amount']) - (current_data['volume'] / prev1_data['volume'])
        else:
            amount_divergence = 0
        
        # Liquidity Regime
        fracture_alignment = volume_gap * amount_break
        
        avg_liquidity = ((prev1_data['volume'] * prev1_data['amount']) + 
                         (prev2_data['volume'] * prev2_data['amount']) + 
                         (prev3_data['volume'] * prev3_data['amount'])) / 3
        if avg_liquidity > 0:
            regime_strength = (current_data['volume'] * current_data['amount']) / avg_liquidity
        else:
            regime_strength = 0
        
        # Microstructure Regime
        # Price Patterns
        prev1_mid = (prev1_data['high'] + prev1_data['low']) / 2
        prev1_range = prev1_data['high'] - prev1_data['low']
        if prev1_range > 0:
            opening_pattern = abs(current_data['open'] - prev1_mid) / prev1_range
        else:
            opening_pattern = 0
            
        current_range = current_data['high'] - current_data['low']
        if current_range > 0:
            intraday_efficiency = (current_data['close'] - current_data['open']) / current_range
        else:
            intraday_efficiency = 0
            
        current_mid = (current_data['high'] + current_data['low']) / 2
        if current_range > 0:
            closing_pressure = abs(current_data['close'] - current_mid) / current_range
        else:
            closing_pressure = 0
        
        # Volume Waves
        if prev1_data['volume'] > 0 and prev2_data['volume'] > 0:
            volume_asymmetry = (current_data['volume'] / prev1_data['volume']) - (prev1_data['volume'] / prev2_data['volume'])
        else:
            volume_asymmetry = 0
            
        if prev1_data['amount'] > 0 and prev2_data['amount'] > 0:
            amount_consistency = (current_data['amount'] / prev1_data['amount']) * (prev1_data['amount'] / prev2_data['amount'])
        else:
            amount_consistency = 0
        
        # Regime Signals
        microstructure_change = opening_pattern * intraday_efficiency * closing_pressure
        volume_break = volume_asymmetry * volume_cluster
        
        # Signal Integration
        # Momentum-Fracture Fusion
        fracture_momentum = momentum_asymmetry * volume_gap
        liquidity_momentum = momentum_convergence * regime_strength
        
        # Regime Weighting
        high_fracture_weight = fracture_alignment * abs(volume_asymmetry)
        low_fracture_weight = 1 / (fracture_alignment * abs(volume_asymmetry) + 1e-6)  # Avoid division by zero
        transition_weight = abs(fracture_alignment - 1) * abs(volume_asymmetry - 1)
        
        # Multi-scale Filtering
        short_term_signal = fracture_momentum * (high_fracture_weight if volume_gap > 0.1 else low_fracture_weight)
        medium_term_signal = volume_momentum * microstructure_change
        
        # Factor Assembly
        # Core Components
        momentum_core = momentum_asymmetry * momentum_convergence
        fracture_core = fracture_alignment * regime_strength
        microstructure_core = microstructure_change * volume_break
        
        # Regime Enhancement
        high_fracture_factor = momentum_core * high_fracture_weight
        low_fracture_factor = fracture_core * low_fracture_weight
        transition_factor = microstructure_core * transition_weight
        
        # Final Factor - weighted combination based on regime conditions
        if volume_gap > 0.15 or amount_break > 0.15:  # High fracture regime
            final_factor = high_fracture_factor + 0.5 * transition_factor
        elif volume_gap < 0.05 and amount_break < 0.05:  # Low fracture regime
            final_factor = low_fracture_factor + 0.3 * transition_factor
        else:  # Transition regime
            final_factor = 0.7 * transition_factor + 0.3 * (high_fracture_factor + low_fracture_factor) / 2
        
        factor.iloc[i] = final_factor
    
    # Fill initial NaN values with 0
    factor = factor.fillna(0)
    
    return factor
