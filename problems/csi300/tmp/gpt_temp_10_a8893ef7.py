import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required intermediate calculations
    df['Bull_Volume_Intensity'] = np.where(df['close'] > df['open'], df['volume'], 0)
    df['Bear_Volume_Intensity'] = np.where(df['close'] < df['open'], df['volume'], 0)
    df['Pressure_Imbalance'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['Range_Utilization_Entropy'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    df['Close_Position_Entropy'] = (df['close'] - (df['high'] + df['low']) / 2) / (df['high'] - df['low']).replace(0, np.nan)
    df['Flow_Distribution'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Volume Spike Regime calculation
    volume_ma = df['volume'].rolling(window=20, min_periods=1).mean()
    df['Volume_Spike_Regime'] = df['volume'] / volume_ma.replace(0, np.nan)
    
    # Tick Efficiency
    df['Tick_Efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Bull/Bear Regime Strength
    bull_volume_ma = df['Bull_Volume_Intensity'].rolling(window=20, min_periods=1).mean()
    bear_volume_ma = df['Bear_Volume_Intensity'].rolling(window=20, min_periods=1).mean()
    df['Bull_Regime_Strength'] = df['Bull_Volume_Intensity'] / bull_volume_ma.replace(0, np.nan)
    df['Bear_Regime_Strength'] = df['Bear_Volume_Intensity'] / bear_volume_ma.replace(0, np.nan)
    
    # Entropy Convergence
    df['Entropy_Convergence'] = df['Range_Utilization_Entropy'] * df['Close_Position_Entropy']
    
    # Asymmetry Ratio
    df['Asymmetry_Ratio'] = df['Bull_Regime_Strength'] / df['Bear_Regime_Strength'].replace(0, np.nan)
    
    # Regime Stability
    df['Regime_Stability'] = 1 / (df['Bull_Regime_Strength'].rolling(window=5).std().replace(0, np.nan) + 
                                df['Bear_Regime_Strength'].rolling(window=5).std().replace(0, np.nan))
    
    for i in range(len(df)):
        if i < 20:
            result.iloc[i] = 0
            continue
            
        try:
            # Multi-Scale Pressure Dynamics
            # Short-Term Pressure Divergence
            bull_vol_intensity_t = df['Bull_Volume_Intensity'].iloc[i]
            bull_vol_intensity_t5 = df['Bull_Volume_Intensity'].iloc[i-5] if i >= 5 else bull_vol_intensity_t
            bear_vol_intensity_t = df['Bear_Volume_Intensity'].iloc[i]
            bear_vol_intensity_t5 = df['Bear_Volume_Intensity'].iloc[i-5] if i >= 5 else bear_voloc_intensity_t
            
            bull_ratio = (bull_vol_intensity_t / bull_vol_intensity_t5 - 1) if bull_vol_intensity_t5 != 0 else 0
            bear_ratio = (bear_vol_intensity_t / bear_vol_intensity_t5 - 1) if bear_vol_intensity_t5 != 0 else 0
            short_term_pressure_divergence = bull_ratio * bear_ratio
            
            # Medium-Term Pressure Divergence
            pressure_imbalance_t = df['Pressure_Imbalance'].iloc[i]
            pressure_imbalance_t20 = df['Pressure_Imbalance'].iloc[i-20]
            amount_t = df['amount'].iloc[i]
            amount_t20 = df['amount'].iloc[i-20]
            
            pressure_ratio = (pressure_imbalance_t / pressure_imbalance_t20 - 1) if pressure_imbalance_t20 != 0 else 0
            amount_ratio = (amount_t / amount_t20 - 1) if amount_t20 != 0 else 0
            medium_term_pressure_divergence = pressure_ratio * amount_ratio
            
            # Pressure Acceleration Asymmetry
            pressure_acceleration_asymmetry = ((short_term_pressure_divergence - medium_term_pressure_divergence) / 
                                             short_term_pressure_divergence) if short_term_pressure_divergence != 0 else 0
            
            # Fractal Range-Pressure Efficiency
            # Short-Term Range Scaling
            current_range = df['high'].iloc[i] - df['low'].iloc[i]
            range_sum = current_range
            for j in range(1, 5):
                if i >= j:
                    range_sum += df['high'].iloc[i-j] - df['low'].iloc[i-j]
            avg_range_5 = range_sum / min(5, i+1)
            short_term_range_scaling = current_range / avg_range_5 if avg_range_5 != 0 else 1
            
            # Pressure Range Efficiency
            pressure_range_efficiency = df['Range_Utilization_Entropy'].iloc[i] * short_term_range_scaling
            
            # Medium-Term Range Scaling
            range_sum_20 = 0
            count = 0
            for j in range(21):
                if i >= j:
                    range_sum_20 += df['high'].iloc[i-j] - df['low'].iloc[i-j]
                    count += 1
            avg_range_20 = range_sum_20 / count if count > 0 else 1
            medium_term_range_scaling = current_range / avg_range_20 if avg_range_20 != 0 else 1
            
            # Range-Pressure Fractality
            range_pressure_fractality = pressure_range_efficiency / medium_term_range_scaling if medium_term_range_scaling != 0 else 0
            
            # Asymmetric Pressure Integration
            pressure_range_composite = pressure_acceleration_asymmetry * range_pressure_fractality
            volume_pressure_fractal = (short_term_pressure_divergence * medium_term_pressure_divergence * 
                                     range_pressure_fractality)
            
            # Microstructure Entropy-Regime Detection
            # Entropy-Order Flow Asymmetry
            opening_entropy_flow = ((df['open'].iloc[i] - df['close'].iloc[i-1]) * 
                                  df['Range_Utilization_Entropy'].iloc[i]) if i >= 1 else 0
            closing_entropy_pressure = (df['Close_Position_Entropy'].iloc[i] * 
                                      (df['close'].iloc[i] - (df['high'].iloc[i] + df['low'].iloc[i]) / 2))
            entropy_flow_asymmetry = opening_entropy_flow - closing_entropy_pressure
            
            # Volume-Entropy Regime Indicators
            volume_spike_entropy = (df['Volume_Spike_Regime'].iloc[i] * df['Tick_Efficiency'].iloc[i])
            
            flow_dist_t = df['Flow_Distribution'].iloc[i]
            flow_dist_t1 = df['Flow_Distribution'].iloc[i-1] if i >= 1 else flow_dist_t
            entropy_concentration_shift = flow_dist_t - flow_dist_t1
            
            range_entropy_t = df['Range_Utilization_Entropy'].iloc[i]
            range_entropy_t1 = df['Range_Utilization_Entropy'].iloc[i-1] if i >= 1 else range_entropy_t
            volume_t = df['volume'].iloc[i]
            volume_t1 = df['volume'].iloc[i-1] if i >= 1 else volume_t
            
            range_entropy_ratio = (range_entropy_t / range_entropy_t1 - 1) if range_entropy_t1 != 0 else 0
            volume_ratio = (volume_t / volume_t1 - 1) if volume_t1 != 0 else 0
            volume_entropy_imbalance = range_entropy_ratio - volume_ratio
            
            # Entropy Break Detection
            entropy_break_intensity = (abs(range_entropy_t - range_entropy_t1) / 
                                     (df['high'].iloc[i-1] - df['low'].iloc[i-1])) if i >= 1 else 0
            volume_entropy_confirmation = volume_spike_entropy * entropy_break_intensity
            entropy_break_signal = entropy_break_intensity * volume_entropy_confirmation
            
            # Asymmetric Entropy-Momentum Construction
            # Fractal Entropy Components
            pressure_imbalance_t3 = df['Pressure_Imbalance'].iloc[i-3] if i >= 3 else pressure_imbalance_t
            high_low_range_3 = 0
            for j in range(4):
                if i >= j:
                    high_low_range_3 += df['high'].iloc[i-j] - df['low'].iloc[i-j]
            high_low_range_3 = high_low_range_3 / min(4, i+1) if min(4, i+1) > 0 else 1
            
            pressure_entropy_asymmetry = ((pressure_imbalance_t - pressure_imbalance_t3) / 
                                        high_low_range_3 * df['Range_Utilization_Entropy'].iloc[i]) if high_low_range_3 != 0 else 0
            
            volume_spike_t3 = df['Volume_Spike_Regime'].iloc[i-3] if i >= 3 else df['Volume_Spike_Regime'].iloc[i]
            volume_entropy_asymmetry = ((df['Volume_Spike_Regime'].iloc[i] - volume_spike_t3) * 
                                      df['Close_Position_Entropy'].iloc[i])
            
            flow_dist_t3 = df['Flow_Distribution'].iloc[i-3] if i >= 3 else flow_dist_t
            flow_entropy_asymmetry = ((flow_dist_t - flow_dist_t3) * 
                                    (df['high'].iloc[i] - df['low'].iloc[i]))
            
            # Cross-Entropy Asymmetry Detection
            pressure_volume_cross = pressure_entropy_asymmetry * volume_entropy_asymmetry
            volume_flow_cross = volume_entropy_asymmetry * flow_entropy_asymmetry
            pressure_flow_cross = pressure_entropy_asymmetry * flow_entropy_asymmetry
            
            # Entropy-Momentum Integration
            triple_entropy_asymmetry = (pressure_entropy_asymmetry * volume_entropy_asymmetry * 
                                      flow_entropy_asymmetry)
            cross_entropy_asymmetry = pressure_volume_cross * volume_flow_cross * pressure_flow_cross
            
            # Entropy-Regime Adaptive Synthesis
            # High Entropy-Volatility Framework
            fractal_entropy_momentum = entropy_break_signal * triple_entropy_asymmetry
            expansion_entropy_efficiency = volume_spike_entropy * pressure_range_composite
            spike_entropy_integration = volume_spike_entropy * cross_entropy_asymmetry
            high_entropy_volatility = fractal_entropy_momentum * expansion_entropy_efficiency * spike_entropy_integration
            
            # Low Entropy-Concentration Framework
            concentration_entropy_momentum = entropy_concentration_shift * cross_entropy_asymmetry
            stability_entropy_efficiency = entropy_concentration_shift * volume_pressure_fractal
            consistency_entropy_integration = entropy_flow_asymmetry * triple_entropy_asymmetry
            low_entropy_concentration = (concentration_entropy_momentum * stability_entropy_efficiency * 
                                       consistency_entropy_integration)
            
            # Entropy-Transition Framework
            entropy_regime_shift = entropy_break_signal * pressure_volume_cross
            volatility_entropy_transition = volume_spike_entropy * flow_entropy_asymmetry
            efficiency_entropy_transition = entropy_flow_asymmetry * volume_entropy_asymmetry
            entropy_transition = entropy_regime_shift * volatility_entropy_transition * efficiency_entropy_transition
            
            # Cross-Entropy Validation & Integration
            # Fractal-Entropy Alignment
            fractal_weighted_entropy = df['Range_Utilization_Entropy'].iloc[i] * range_pressure_fractality
            entropy_fractal_momentum = fractal_weighted_entropy * volume_spike_entropy
            flow_entropy_fractal = df['Flow_Distribution'].iloc[i] * df['Entropy_Convergence'].iloc[i]
            
            # Pressure-Entropy Fractal Integration
            pressure_entropy_fractal = df['Pressure_Imbalance'].iloc[i] * entropy_fractal_momentum
            entropy_pressure_fractal_alignment = (pressure_entropy_fractal * df['Asymmetry_Ratio'].iloc[i])
            regime_entropy_fractal_validation = (entropy_pressure_fractal_alignment * 
                                               df['Regime_Stability'].iloc[i])
            
            # Fractal Asymmetry Validation
            bull_bear_fractal_entropy = (df['Bull_Regime_Strength'].iloc[i] * fractal_entropy_momentum)
            bear_bull_fractal_entropy = (df['Bear_Regime_Strength'].iloc[i] * consistency_entropy_integration)
            net_fractal_asymmetry = bull_bear_fractal_entropy - bear_bull_fractal_entropy
            
            # Composite Alpha Generation
            # Core Fractal-Entropy Components
            fractal_pressure_entropy_factor = pressure_range_composite * volume_pressure_fractal
            entropy_regime_momentum_factor = (high_entropy_volatility * low_entropy_concentration * 
                                            entropy_transition)
            microstructure_entropy_factor = entropy_flow_asymmetry * volume_entropy_imbalance
            
            # Dynamic Entropy-Weighting Scheme
            entropy_volatility_weight = volume_spike_entropy * entropy_break_signal
            entropy_efficiency_weight = entropy_flow_asymmetry * entropy_concentration_shift
            entropy_momentum_weight = pressure_acceleration_asymmetry * range_pressure_fractality
            
            # Final Alpha Construction
            base_fractal_entropy_alpha = (fractal_pressure_entropy_factor * entropy_regime_momentum_factor * 
                                        microstructure_entropy_factor)
            
            adaptive_weight = (1 + entropy_volatility_weight * entropy_efficiency_weight * 
                             entropy_momentum_weight)
            
            final_alpha = base_fractal_entropy_alpha * adaptive_weight
            
            result.iloc[i] = final_alpha
            
        except (ZeroDivisionError, ValueError, IndexError):
            result.iloc[i] = 0
    
    return result
