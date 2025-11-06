import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate basic components
    df['TrueRange'] = np.maximum(df['high'] - df['low'], 
                                np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                          abs(df['low'] - df['close'].shift(1))))
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 20:  # Need at least 20 days of data
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]
        
        # Multi-Scale Entropy-Efficiency Confluence
        # Price Entropy-Efficiency Components
        close_prices = current_data['close'].values
        
        # 5-day price entropy
        if i >= 4:
            price_changes_5 = np.abs(np.diff(close_prices[i-4:i+1]))
            if np.sum(price_changes_5) > 0:
                p_i_5 = price_changes_5 / np.sum(price_changes_5)
                entropy_5 = -np.sum(p_i_5 * np.log(p_i_5 + 1e-10))
            else:
                entropy_5 = 0
        else:
            entropy_5 = 0
            
        # 10-day price entropy
        if i >= 9:
            price_changes_10 = np.abs(np.diff(close_prices[i-9:i+1]))
            if np.sum(price_changes_10) > 0:
                p_i_10 = price_changes_10 / np.sum(price_changes_10)
                entropy_10 = -np.sum(p_i_10 * np.log(p_i_10 + 1e-10))
            else:
                entropy_10 = 0
        else:
            entropy_10 = 0
            
        # Movement efficiencies
        if i >= 4:
            movement_sum_5 = np.sum(np.abs(np.diff(close_prices[i-4:i+1])))
            tr_sum_5 = np.sum(current_data['TrueRange'].values[i-4:i+1])
            efficiency_5 = movement_sum_5 / tr_sum_5 if tr_sum_5 > 0 else 0
        else:
            efficiency_5 = 0
            
        if i >= 9:
            movement_sum_10 = np.sum(np.abs(np.diff(close_prices[i-9:i+1])))
            tr_sum_10 = np.sum(current_data['TrueRange'].values[i-9:i+1])
            efficiency_10 = movement_sum_10 / tr_sum_10 if tr_sum_10 > 0 else 0
        else:
            efficiency_10 = 0
            
        entropy_efficiency_divergence = (entropy_10 - entropy_5) * (efficiency_5 / (efficiency_10 + 1e-10) - 1)
        
        # Volume Efficiency Dynamics
        volumes = current_data['volume'].values
        
        # Volume entropy
        if i >= 4:
            vol_5 = volumes[i-4:i+1]
            if np.sum(vol_5) > 0:
                q_i_5 = vol_5 / np.sum(vol_5)
                vol_entropy_5 = -np.sum(q_i_5 * np.log(q_i_5 + 1e-10))
            else:
                vol_entropy_5 = 0
        else:
            vol_entropy_5 = 0
            
        if i >= 9:
            vol_10 = volumes[i-9:i+1]
            if np.sum(vol_10) > 0:
                q_i_10 = vol_10 / np.sum(vol_10)
                vol_entropy_10 = -np.sum(q_i_10 * np.log(q_i_10 + 1e-10))
            else:
                vol_entropy_10 = 0
        else:
            vol_entropy_10 = 0
            
        # Intraday efficiency and volume ratio
        high, low, open_price, close_price = current_data.iloc[i][['high', 'low', 'open', 'close']]
        intraday_efficiency = (close_price - open_price) / (high - low + 1e-10)
        
        if i >= 4:
            avg_vol_prev = np.mean(volumes[i-4:i])
            volume_ratio = volumes[i] / (avg_vol_prev + 1e-10)
        else:
            volume_ratio = 1
            
        volume_efficiency_gradient = (vol_entropy_10 - vol_entropy_5) * (intraday_efficiency * volume_ratio)
        
        # Entropy-Efficiency Quality Score
        entropy_efficiency_quality = entropy_efficiency_divergence * volume_efficiency_gradient
        entropy_efficiency_quality = np.tanh(entropy_efficiency_quality)
        
        # Fractal Momentum with Efficiency Weighting
        # Multi-Scale Momentum Fractals
        if i >= 3:
            fractal_3 = (close_prices[i] / close_prices[i-2] - 1) * (close_prices[i-1] / close_prices[i-3] - 1)
        else:
            fractal_3 = 0
            
        if i >= 4:
            momentum_5 = close_prices[i] / close_prices[i-4] - 1
        else:
            momentum_5 = 0
            
        if i >= 19:
            momentum_20 = close_prices[i] / close_prices[i-19] - 1
        else:
            momentum_20 = 0
            
        fractal_consistency = 1 if np.sign(fractal_3) == np.sign(momentum_5) else 0
        fractal_momentum_divergence = (momentum_5 - momentum_20) * fractal_consistency
        
        # Efficiency-Weighted Regime Detection
        high_efficiency_regime = efficiency_5 > 1.2 * efficiency_10 if efficiency_10 > 0 else False
        volume_confirmation_regime = volumes[i] > 2 * avg_vol_prev and volume_ratio > 1.5 if i >= 4 else False
        intraday_efficiency_regime = abs(intraday_efficiency) > 0.6
        entropy_stability_regime = abs(entropy_10 - entropy_5) < 0.3
        
        # Efficiency-Weighted Momentum Scoring
        base_momentum = np.mean([fractal_3, momentum_5]) if i >= 4 else momentum_5
        
        if high_efficiency_regime:
            base_momentum *= 1.4
            
        if volume_confirmation_regime:
            base_momentum *= volume_ratio
            
        if not intraday_efficiency_regime and fractal_consistency < 2:
            base_momentum = 0
            
        if entropy_stability_regime:
            base_momentum *= 1.2
            
        fractal_momentum_score = base_momentum
        
        # Volume-Price Efficiency Confluence
        price_momentum_divergence = momentum_5 - momentum_20
        
        # Volume-Price Correlation
        if i >= 2:
            recent_data = current_data.iloc[i-2:i+1]
            volume_price_corr = np.corrcoef(recent_data['close'], recent_data['volume'])[0,1]
            if np.isnan(volume_price_corr):
                volume_price_corr = 0
        else:
            volume_price_corr = 0
            
        # VWP Efficiency
        if i >= 9:
            current_vwp = np.sum(close_prices[i-4:i+1] * volumes[i-4:i+1]) / np.sum(volumes[i-4:i+1])
            prev_vwp = np.sum(close_prices[i-9:i-4] * volumes[i-9:i-4]) / np.sum(volumes[i-9:i-4])
            vwp_efficiency = (current_vwp - prev_vwp) / (abs(prev_vwp) + 1e-10) * efficiency_5
        else:
            vwp_efficiency = 0
            
        # Intraday Pressure Efficiency
        intraday_pressure = (high - open_price) - (open_price - low)
        pressure_efficiency = intraday_pressure / (high - low + 1e-10) * volume_ratio
        
        # Efficiency Alignment Patterns
        momentum_correlation_alignment = np.sign(price_momentum_divergence) * np.sign(volume_price_corr) * min(abs(price_momentum_divergence), abs(volume_price_corr))
        vwp_pressure_correlation = np.sign(vwp_efficiency) * np.sign(pressure_efficiency)
        
        # Multi-Efficiency Consistency
        efficiency_components = [price_momentum_divergence, volume_price_corr, vwp_efficiency, pressure_efficiency]
        same_sign_count = sum(1 for j in range(len(efficiency_components)-1) 
                            for k in range(j+1, len(efficiency_components)) 
                            if np.sign(efficiency_components[j]) == np.sign(efficiency_components[k]))
        
        # Efficiency Confluence Score
        efficiency_confluence = momentum_correlation_alignment * vwp_pressure_correlation * same_sign_count
        efficiency_confluence = efficiency_confluence ** 3
        
        # Liquidity-Efficiency Quality Assessment
        movement_efficiency = abs(close_prices[i] - close_prices[i-1]) / current_data['TrueRange'].values[i]
        
        if i >= 4:
            avg_movement_efficiency = np.mean([abs(close_prices[j] - close_prices[j-1]) / current_data['TrueRange'].values[j] 
                                             for j in range(i-4, i)])
            efficiency_quality_ratio = movement_efficiency / (avg_movement_efficiency + 1e-10)
        else:
            efficiency_quality_ratio = 1
            
        volume_asymmetry = volume_ratio
        
        # Efficiency Momentum Patterns
        if i >= 8:
            efficiency_momentum = movement_efficiency / (abs(close_prices[i-4] - close_prices[i-5]) / current_data['TrueRange'].values[i-4] + 1e-10) - 1
            efficiency_volatility = np.std([abs(close_prices[j] - close_prices[j-1]) / current_data['TrueRange'].values[j] 
                                          for j in range(i-9, i+1)])
            
            # Efficiency persistence
            efficiency_changes = []
            for j in range(i-4, i+1):
                if j > 0:
                    curr_eff = abs(close_prices[j] - close_prices[j-1]) / current_data['TrueRange'].values[j]
                    prev_eff = abs(close_prices[j-1] - close_prices[j-2]) / current_data['TrueRange'].values[j-1]
                    efficiency_changes.append(np.sign(curr_eff - prev_eff))
            
            efficiency_persistence = 0
            if efficiency_changes:
                current_sign = efficiency_changes[-1]
                count = 0
                for sign in reversed(efficiency_changes):
                    if sign == current_sign:
                        count += 1
                    else:
                        break
                efficiency_persistence = count
        else:
            efficiency_momentum = 0
            efficiency_volatility = 1
            efficiency_persistence = 1
            
        efficiency_momentum_score = efficiency_momentum * efficiency_persistence / (efficiency_volatility + 0.001)
        
        # Liquidity-Efficiency Quality Score
        liquidity_efficiency_quality = efficiency_quality_ratio * volume_asymmetry * efficiency_momentum_score
        
        # Adaptive Multi-Fractal Efficiency Signal Generation
        # Regime-Weighted Component Integration
        if high_efficiency_regime:
            entropy_weight = 0.8
            fractal_weight = 1.5
            confluence_weight = 1.4
            liquidity_weight = 1.0
        elif volume_confirmation_regime:
            entropy_weight = 1.0
            fractal_weight = 0.9
            confluence_weight = 1.3
            liquidity_weight = 1.6
        else:
            entropy_weight = 1.2
            fractal_weight = 1.0
            confluence_weight = 1.0
            liquidity_weight = 1.0
            
        # Dynamic Component Multiplication
        signal = (entropy_efficiency_quality * entropy_weight) * \
                (fractal_momentum_score * fractal_weight) * \
                (efficiency_confluence * confluence_weight) * \
                (liquidity_efficiency_quality * liquidity_weight)
        
        # Signal Quality Validation
        fractal_consistency_filter = fractal_consistency >= 1
        entropy_stability_filter = abs(entropy_10 - entropy_5) < 0.4
        efficiency_quality_filter = efficiency_quality_ratio > 0.7
        efficiency_alignment_filter = same_sign_count >= 2
        
        if not (fractal_consistency_filter and entropy_stability_filter and 
                efficiency_quality_filter and efficiency_alignment_filter):
            signal *= 0.5  # Reduce signal strength for lower quality
            
        # Final Signal Refinement
        signal = np.cbrt(signal)
        
        # Apply regime-specific final adjustments
        if high_efficiency_regime:
            signal *= 1.2
        if volume_confirmation_regime:
            signal *= 1.3
            
        # Incorporate intraday pressure efficiency and VWP efficiency
        signal *= (1 + 0.1 * pressure_efficiency)
        signal *= (1 + 0.1 * vwp_efficiency)
        
        result.iloc[i] = signal
        
    return result
