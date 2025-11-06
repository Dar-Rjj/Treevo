import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(5, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # Multi-Timeframe Fractal Analysis
        # Price Fractal Components
        intraday_price_fractal = (current_data['close'].iloc[-1] - current_data['open'].iloc[-1]) / (current_data['high'].iloc[-1] - current_data['low'].iloc[-1] + 1e-8)
        overnight_price_fractal = (current_data['open'].iloc[-1] - current_data['close'].iloc[-2]) / (current_data['high'].iloc[-1] - current_data['low'].iloc[-1] + 1e-8)
        gap_volatility_ratio = abs(current_data['open'].iloc[-1] - current_data['close'].iloc[-2]) / (current_data['high'].iloc[-1] - current_data['low'].iloc[-1] + 1e-8)
        
        # Volume Fractal Components
        vol_sum_3d = current_data['volume'].iloc[-4:-1].sum() + 1e-8
        volume_concentration_fractal = current_data['volume'].iloc[-1] / vol_sum_3d
        liquidity_depth_fractal = current_data['volume'].iloc[-1] / (abs(current_data['close'].iloc[-1] - current_data['open'].iloc[-1]) + 0.0001)
        volume_momentum_fractal = (current_data['volume'].iloc[-1] - current_data['volume'].iloc[-2]) / vol_sum_3d
        
        # Fractal Interaction Dynamics
        price_volume_fractal_corr = intraday_price_fractal * volume_momentum_fractal
        gap_volume_fractal_alignment = gap_volatility_ratio * volume_concentration_fractal
        fractal_persistence = np.sign(intraday_price_fractal) * np.sign(overnight_price_fractal)
        
        # Volatility-Regime Resonance Detection
        # Volatility Fractal Classification
        if gap_volatility_ratio > 0.7 and intraday_price_fractal > 0.3:
            volatility_regime = 2  # High volatility
        elif gap_volatility_ratio < 0.3 and abs(intraday_price_fractal) < 0.2:
            volatility_regime = 0  # Low volatility
        else:
            volatility_regime = 1  # Normal volatility
        
        # Volume Fractal Confirmation
        if volume_concentration_fractal > 0.4:
            volume_regime = 2  # Volume clustering
        elif volume_concentration_fractal < 0.2:
            volume_regime = 0  # Volume dispersion
        else:
            volume_regime = 1  # Volume stability
        
        # Fractal Regime Interaction
        volatility_volume_alignment = volatility_regime * volume_regime
        
        # Calculate regime persistence (last 3 days including current)
        regime_combinations = []
        for j in range(max(0, i-2), i+1):
            gap_ratio_j = abs(current_data['open'].iloc[j] - current_data['close'].iloc[j-1]) / (current_data['high'].iloc[j] - current_data['low'].iloc[j] + 1e-8)
            intraday_j = (current_data['close'].iloc[j] - current_data['open'].iloc[j]) / (current_data['high'].iloc[j] - current_data['low'].iloc[j] + 1e-8)
            vol_conc_j = current_data['volume'].iloc[j] / (current_data['volume'].iloc[j-3:j].sum() + 1e-8)
            
            if gap_ratio_j > 0.7 and intraday_j > 0.3:
                vol_regime_j = 2
            elif gap_ratio_j < 0.3 and abs(intraday_j) < 0.2:
                vol_regime_j = 0
            else:
                vol_regime_j = 1
                
            if vol_conc_j > 0.4:
                vol_vol_regime_j = 2
            elif vol_conc_j < 0.2:
                vol_vol_regime_j = 0
            else:
                vol_vol_regime_j = 1
                
            regime_combinations.append((vol_regime_j, vol_vol_regime_j))
        
        current_regime = (volatility_regime, volume_regime)
        regime_persistence = sum(1 for regime in regime_combinations[-3:] if regime == current_regime)
        regime_transition_prob = 1 / (1 + regime_persistence)
        
        # Momentum-Liquidity Fractal Construction
        # Core Momentum Fractals
        high_range_2d = current_data['high'].iloc[-3:].max() - current_data['low'].iloc[-3:].min()
        short_term_momentum_fractal = (current_data['close'].iloc[-1] - current_data['close'].iloc[-3]) / (high_range_2d + 1e-8)
        
        high_range_5d = current_data['high'].iloc[-6:].max() - current_data['low'].iloc[-6:].min()
        medium_term_momentum_fractal = (current_data['close'].iloc[-1] - current_data['close'].iloc[-6]) / (high_range_5d + 1e-8)
        
        momentum_curvature_fractal = (current_data['close'].iloc[-1] - 2 * current_data['close'].iloc[-2] + current_data['close'].iloc[-3]) / (current_data['high'].iloc[-1] - current_data['low'].iloc[-1] + 1e-8)
        
        # Liquidity Enhancement Fractals
        liquidity_depth_prev = current_data['volume'].iloc[-2] / (abs(current_data['close'].iloc[-2] - current_data['open'].iloc[-2]) + 0.0001)
        liquidity_momentum_fractal = liquidity_depth_fractal / (liquidity_depth_prev + 1e-8) - 1
        
        volume_pressure_fractal = (current_data['close'].iloc[-1] - current_data['open'].iloc[-1]) / (current_data['volume'].iloc[-1] + 0.0001) * volume_concentration_fractal
        
        vwap_t = current_data['amount'].iloc[-1] / (current_data['volume'].iloc[-1] + 1e-8)
        price_impact_fractal = (current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) / (vwap_t + 1e-8)
        
        # Fractal Momentum Integration
        momentum_liquidity_product = short_term_momentum_fractal * liquidity_momentum_fractal
        volume_confirmed_momentum = medium_term_momentum_fractal * volume_pressure_fractal
        fractal_momentum_resonance = momentum_curvature_fractal * price_impact_fractal
        
        # Regime-Adaptive Fractal Fusion
        # High Volatility Regime Factors
        volatility_capture_fractal = gap_volatility_ratio * momentum_liquidity_product
        volume_stability_fractal = volatility_capture_fractal / (1 + abs(volume_momentum_fractal))
        high_volatility_alpha = volume_stability_fractal * fractal_momentum_resonance
        
        # Low Volatility Regime Factors
        momentum_resonance_fractal = momentum_liquidity_product * volume_confirmed_momentum
        liquidity_confirmation_fractal = momentum_resonance_fractal * liquidity_depth_fractal
        low_volatility_alpha = liquidity_confirmation_fractal * fractal_persistence
        
        # Normal Volatility Regime Factors
        balanced_momentum_fractal = (momentum_liquidity_product + volume_confirmed_momentum) / 2
        volume_price_alignment = balanced_momentum_fractal * price_volume_fractal_corr
        normal_volatility_alpha = volume_price_alignment * gap_volume_fractal_alignment
        
        # Dynamic Fractal Blending
        # Regime Weight Calculation
        high_volatility_weight = gap_volatility_ratio * regime_transition_prob
        low_volatility_weight = (1 - gap_volatility_ratio) * regime_transition_prob
        normal_volatility_weight = abs(intraday_price_fractal) * regime_transition_prob
        
        weight_normalization = 1 / (high_volatility_weight + low_volatility_weight + normal_volatility_weight + 1e-8)
        
        # Weighted Factor Fusion
        weighted_high_volatility = high_volatility_alpha * high_volatility_weight * weight_normalization
        weighted_low_volatility = low_volatility_alpha * low_volatility_weight * weight_normalization
        weighted_normal_volatility = normal_volatility_alpha * normal_volatility_weight * weight_normalization
        
        fused_fractal_factor = weighted_high_volatility + weighted_low_volatility + weighted_normal_volatility
        
        # Fractal Validation
        volume_flow_persistence = np.sign(current_data['volume'].iloc[-1] - current_data['volume'].iloc[-2]) * np.sign(current_data['volume'].iloc[-2] - current_data['volume'].iloc[-3])
        
        price_changes = []
        for j in range(max(0, i-4), i+1):
            if j > 0:
                price_changes.append(np.sign(current_data['close'].iloc[j] - current_data['close'].iloc[j-1]))
        price_pattern_consistency = len(price_changes) if price_changes else 1
        
        validated_fractal_factor = fused_fractal_factor * volume_flow_persistence * price_pattern_consistency
        
        # Final Alpha Generation
        core_fractal_alpha = validated_fractal_factor * fractal_momentum_resonance
        microstructure_refinement = core_fractal_alpha * price_impact_fractal
        final_alpha = microstructure_refinement * liquidity_depth_fractal
        
        result.iloc[i] = final_alpha
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
