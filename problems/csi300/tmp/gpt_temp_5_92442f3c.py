import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize all intermediate columns
    data['prev_close'] = data['close'].shift(1)
    data['prev_net_asym'] = np.nan
    data['prev_volume_efficiency'] = np.nan
    data['prev_price_trade_intensity'] = np.nan
    data['prev_efficiency_exec_quality'] = np.nan
    
    # Calculate intermediate components
    results = []
    
    for i in range(len(data)):
        if i < 1:
            # Skip first row due to lag requirements
            results.append(np.nan)
            continue
            
        # Current values
        open_t = data['open'].iloc[i]
        high_t = data['high'].iloc[i]
        low_t = data['low'].iloc[i]
        close_t = data['close'].iloc[i]
        volume_t = data['volume'].iloc[i]
        amount_t = data['amount'].iloc[i]
        prev_close = data['prev_close'].iloc[i]
        
        # 1. Asymmetric Momentum Analysis
        # Upside momentum components
        upper_momentum = (high_t - open_t) / open_t * volume_t if close_t > data['close'].iloc[i-1] else 0
        lower_momentum = (open_t - low_t) / open_t * volume_t if close_t < data['close'].iloc[i-1] else 0
        
        # Net asymmetric momentum
        net_asym_momentum = (upper_momentum - lower_momentum) / (upper_momentum + lower_momentum + 1e-6)
        
        # Multi-scale asymmetry
        if i >= 1:
            prev_net_asym = data['prev_net_asym'].iloc[i-1] if not np.isnan(data['prev_net_asym'].iloc[i-1]) else 0
            one_day_asym_change = net_asym_momentum - prev_net_asym
        else:
            one_day_asym_change = 0
            
        if i >= 3:
            net_asym_3 = data['prev_net_asym'].iloc[i-3] if not np.isnan(data['prev_net_asym'].iloc[i-3]) else 0
            three_day_asym_div = net_asym_momentum - net_asym_3
        else:
            three_day_asym_div = 0
            
        asymmetry_consistency = np.sign(one_day_asym_change) * np.sign(three_day_asym_div)
        
        # Efficiency-enhanced asymmetry
        range_eff_denom = max(high_t - low_t, abs(high_t - prev_close), abs(low_t - prev_close))
        range_efficiency = volume_t / (range_eff_denom + 1e-6)
        gap_efficiency = (close_t - open_t) / (abs(open_t - prev_close) + 1e-6)
        efficiency_composite = range_efficiency * gap_efficiency
        
        # 2. Volume-Asymmetry Integration
        # Volume efficiency dynamics
        volume_efficiency_intensity = volume_t / (open_t + 1e-6)
        
        if i >= 1:
            prev_vol_eff = data['prev_volume_efficiency'].iloc[i-1] if not np.isnan(data['prev_volume_efficiency'].iloc[i-1]) else volume_efficiency_intensity
            volume_efficiency_momentum = volume_efficiency_intensity / (prev_vol_eff + 1e-6) - 1
        else:
            volume_efficiency_momentum = 0
            
        if i >= 2:
            prev_vol_mom = data['prev_volume_efficiency'].iloc[i-2] if not np.isnan(data['prev_volume_efficiency'].iloc[i-2]) else volume_efficiency_intensity
            prev_vol_eff_current = data['prev_volume_efficiency'].iloc[i-1] if not np.isnan(data['prev_volume_efficiency'].iloc[i-1]) else volume_efficiency_intensity
            prev_vol_momentum = prev_vol_eff_current / (prev_vol_mom + 1e-6) - 1
            volume_efficiency_acceleration = volume_efficiency_momentum - prev_vol_momentum
        else:
            volume_efficiency_acceleration = 0
            
        # Asymmetry-volume alignment
        direction_alignment = np.sign(volume_efficiency_momentum) * np.sign(net_asym_momentum)
        magnitude_divergence = volume_efficiency_momentum - net_asym_momentum
        acceleration_coherence = volume_efficiency_acceleration * one_day_asym_change
        
        # Trade size asymmetry
        price_trade_intensity = (amount_t / (volume_t + 1e-6)) * net_asym_momentum
        
        if i >= 1:
            prev_price_intensity = data['prev_price_trade_intensity'].iloc[i-1] if not np.isnan(data['prev_price_trade_intensity'].iloc[i-1]) else price_trade_intensity
            size_asymmetry_momentum = price_trade_intensity / (prev_price_intensity + 1e-6) - 1
        else:
            size_asymmetry_momentum = 0
            
        size_volume_asymmetry = size_asymmetry_momentum * volume_efficiency_momentum
        
        # 3. Multi-Scale Efficiency Regime Detection
        # Efficiency momentum regimes
        high_efficiency_regime = ((high_t - low_t) / (open_t + 1e-6)) * net_asym_momentum
        low_efficiency_regime = (1 - (high_t - low_t) / (open_t + 1e-6)) * net_asym_momentum
        
        if i >= 2:
            net_asym_2 = data['prev_net_asym'].iloc[i-2] if not np.isnan(data['prev_net_asym'].iloc[i-2]) else 0
            efficiency_regime_shift = np.sign(net_asym_momentum - net_asym_2)
        else:
            efficiency_regime_shift = 0
            
        # Volume-efficiency regime interaction
        high_volume_efficiency = volume_efficiency_intensity * high_efficiency_regime
        low_volume_efficiency = volume_efficiency_intensity * low_efficiency_regime
        volume_regime_alignment = volume_efficiency_momentum * efficiency_regime_shift
        
        # Multi-timeframe efficiency strength
        short_term_efficiency_persistence = 1 if net_asym_momentum > 0 else 0
        
        if i >= 4:
            net_asym_4 = data['prev_net_asym'].iloc[i-4] if not np.isnan(data['prev_net_asym'].iloc[i-4]) else 0
            medium_term_efficiency_momentum = net_asym_momentum / (net_asym_4 + 1e-6) - 1
        else:
            medium_term_efficiency_momentum = 0
            
        efficiency_strength_alignment = efficiency_regime_shift * volume_regime_alignment
        
        # 4. Microstructure Efficiency Anchoring
        # Opening efficiency dynamics
        gap_momentum_efficiency = ((open_t - prev_close) / (prev_close + 1e-6)) * (volume_t / (open_t + 1e-6))
        opening_eff_pressure_denom = max(abs(open_t - prev_close), abs(high_t - open_t), abs(low_t - open_t))
        opening_efficiency_pressure = volume_t / (opening_eff_pressure_denom + 1e-6)
        gap_efficiency_reversal = np.sign(open_t - prev_close) * np.sign(close_t - open_t) * net_asym_momentum
        
        # Intraday efficiency patterns
        range_eff_util_denom = (high_t - low_t) + abs(high_t - prev_close) + abs(low_t - prev_close)
        range_efficiency_utilization = (high_t - low_t) / (range_eff_util_denom + 1e-6)
        close_efficiency_ratio = ((close_t - open_t) / ((high_t - low_t) + 1e-6)) * (amount_t / (volume_t + 1e-6))
        micro_efficiency_momentum = ((close_t - open_t) / ((high_t - low_t) + 1e-6)) * (volume_t / (open_t + 1e-6))
        
        # Session efficiency coherence
        opening_closing_efficiency = gap_momentum_efficiency * close_efficiency_ratio
        intraday_efficiency_quality = range_efficiency_utilization * micro_efficiency_momentum
        session_efficiency_persistence = opening_closing_efficiency * intraday_efficiency_quality
        
        # 5. Asymmetric Flow Quality Assessment
        # Execution efficiency quality
        price_impact_efficiency = ((close_t - open_t) / ((high_t - low_t) + 1e-6)) * volume_efficiency_intensity
        microstructure_efficiency_ratio = net_asym_momentum * (volume_t / (amount_t + 1e-6))
        efficiency_execution_quality = price_impact_efficiency * microstructure_efficiency_ratio
        
        # Flow quality efficiency signals
        high_quality_efficiency_flow = net_asym_momentum * volume_efficiency_intensity * efficiency_composite
        quality_efficiency_transition = volume_efficiency_momentum * net_asym_momentum * ((close_t - open_t) / (open_t + 1e-6))
        efficiency_stability_metric = efficiency_execution_quality * session_efficiency_persistence
        
        # Multi-scale quality efficiency
        if i >= 1:
            prev_eff_exec_quality = data['prev_efficiency_exec_quality'].iloc[i-1] if not np.isnan(data['prev_efficiency_exec_quality'].iloc[i-1]) else efficiency_execution_quality
            short_term_quality_efficiency = efficiency_execution_quality - prev_eff_exec_quality
        else:
            short_term_quality_efficiency = 0
            
        quality_regime_efficiency = efficiency_execution_quality * efficiency_regime_shift
        quality_volume_efficiency = efficiency_execution_quality * volume_efficiency_momentum
        
        # 6. Composite Efficiency Alpha Generation
        # Core asymmetric efficiency engine
        directional_efficiency_momentum = high_efficiency_regime * low_efficiency_regime
        volume_confirmed_efficiency = directional_efficiency_momentum * direction_alignment
        multi_scale_efficiency_validation = volume_confirmed_efficiency * asymmetry_consistency
        
        # Microstructure efficiency enhancement
        quality_weighted_efficiency = multi_scale_efficiency_validation * efficiency_execution_quality
        session_timing_efficiency = quality_weighted_efficiency * session_efficiency_persistence
        regime_adaptive_efficiency = session_timing_efficiency * efficiency_regime_shift
        
        # Final alpha components
        divergence_enhanced_efficiency = regime_adaptive_efficiency * magnitude_divergence
        microstructure_validated_efficiency = divergence_enhanced_efficiency * opening_closing_efficiency
        
        # Final composite alpha
        final_alpha = (0.4 * divergence_enhanced_efficiency + 
                      0.35 * microstructure_validated_efficiency + 
                      0.25 * quality_volume_efficiency)
        
        results.append(final_alpha)
        
        # Store current values for next iteration
        data.loc[data.index[i], 'prev_net_asym'] = net_asym_momentum
        data.loc[data.index[i], 'prev_volume_efficiency'] = volume_efficiency_intensity
        data.loc[data.index[i], 'prev_price_trade_intensity'] = price_trade_intensity
        data.loc[data.index[i], 'prev_efficiency_exec_quality'] = efficiency_execution_quality
    
    return pd.Series(results, index=data.index)
