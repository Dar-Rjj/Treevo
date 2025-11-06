import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 5:  # Need enough history for calculations
            result.iloc[i] = 0
            continue
            
        current = df.iloc[i]
        prev_1 = df.iloc[i-1]
        prev_3 = df.iloc[i-3] if i >= 3 else None
        prev_4 = df.iloc[i-4] if i >= 4 else None
        
        # Extract current values
        open_t = current['open']
        high_t = current['high']
        low_t = current['low']
        close_t = current['close']
        amount_t = current['amount']
        volume_t = current['volume']
        
        # Previous values
        close_t_1 = prev_1['close']
        volume_t_1 = prev_1['volume']
        amount_t_1 = prev_1['amount']
        
        # Calculate basic components
        price_range = high_t - low_t
        if price_range == 0:
            price_range = 1e-10
            
        # 1. Bidirectional Flow Imbalance Analysis
        buy_pressure = (close_t - low_t) * volume_t / price_range
        sell_pressure = (high_t - close_t) * volume_t / price_range
        flow_imbalance = (buy_pressure - sell_pressure) / (buy_pressure + sell_pressure + 1e-10)
        
        # 2. Cross-Scale Volume Distribution
        large_trade_dominance = amount_t / (volume_t + 1e-10)
        large_trade_dominance_prev = amount_t_1 / (volume_t_1 + 1e-10)
        volume_concentration_shift = large_trade_dominance / (large_trade_dominance_prev + 1e-10)
        
        # Calculate 5-day average of Amount/Volume
        amount_volume_ratio_5d = []
        for j in range(max(0, i-4), i+1):
            if j < len(df):
                amount_volume_ratio_5d.append(df.iloc[j]['amount'] / (df.iloc[j]['volume'] + 1e-10))
        trade_size_momentum_div = large_trade_dominance - np.mean(amount_volume_ratio_5d)
        
        # 3. Microstructure Absorption Patterns
        opening_absorption = abs(open_t - close_t_1) / price_range
        intraday_absorption = (close_t - open_t) ** 2 / price_range
        absorption_asymmetry = opening_absorption * intraday_absorption
        
        # 4. Cross-Timeframe Signal Coherence
        short_term_info_density = abs(close_t - close_t_1) / price_range
        
        # Medium-term information persistence
        close_prices_3d = [df.iloc[j]['close'] for j in range(max(0, i-2), i+1)]
        close_changes_3d = [close_prices_3d[k] - close_prices_3d[k-1] for k in range(1, len(close_prices_3d))]
        if len(close_changes_3d) >= 2:
            std_3d = np.std(close_prices_3d)
            mean_abs_changes = np.mean([abs(chg) for chg in close_changes_3d])
            medium_term_persistence = std_3d / (mean_abs_changes + 1e-10)
        else:
            medium_term_persistence = 1.0
            
        info_coherence_ratio = short_term_info_density / (medium_term_persistence + 1e-10)
        
        # 5. Volume-Information Synchronization
        volume_informed_moves = np.sign(close_t - close_t_1) * (volume_t / (volume_t_1 + 1e-10))
        info_efficiency_score = (abs(close_t - open_t) / price_range) * (volume_t / (volume_t_1 + 1e-10))
        synchronization_momentum = volume_informed_moves * info_efficiency_score
        
        # 6. Regime-Dependent Information Flow
        high_info_regime = info_coherence_ratio if volume_t > 1.5 * volume_t_1 else 0
        low_info_regime = synchronization_momentum if volume_t < 0.8 * volume_t_1 else 0
        info_regime_transition = high_info_regime - low_info_regime
        
        # 7. Nonlinear Price Impact Dynamics
        price_impact_curvature = (close_t - open_t) ** 2 / price_range
        volume_impact_nonlinearity = abs(close_t - open_t) * volume_t / price_range
        impact_convexity_score = price_impact_curvature * volume_impact_nonlinearity
        
        # 8. Asymmetric Liquidity Provision
        bid_liquidity = ((close_t - low_t) / price_range) * volume_t
        ask_liquidity = ((high_t - close_t) / price_range) * volume_t
        liquidity_asymmetry = (bid_liquidity - ask_liquidity) / (bid_liquidity + ask_liquidity + 1e-10)
        
        # 9. Dynamic Impact Regimes
        impact_convexity_5d = []
        for j in range(max(0, i-4), i+1):
            if j < len(df):
                curr = df.iloc[j]
                price_range_j = curr['high'] - curr['low']
                if price_range_j == 0:
                    price_range_j = 1e-10
                impact_curv = (curr['close'] - curr['open']) ** 2 / price_range_j
                vol_impact = abs(curr['close'] - curr['open']) * curr['volume'] / price_range_j
                impact_convexity_5d.append(impact_curv * vol_impact)
        
        impact_median = np.median(impact_convexity_5d) if impact_convexity_5d else 0
        
        high_impact_efficiency = info_efficiency_score if impact_convexity_score > impact_median else 0
        low_impact_persistence = flow_imbalance if impact_convexity_score < impact_median else 0
        impact_regime_factor = high_impact_efficiency * low_impact_persistence
        
        # 10. Multi-Frequency Momentum Coupling
        high_freq_momentum = (close_t - close_t_1) / price_range
        
        # Low-frequency momentum
        if i >= 3:
            close_t_3 = df.iloc[i-3]['close']
            high_prices_3d = [df.iloc[j]['high'] for j in range(max(0, i-2), i+1)]
            low_prices_3d = [df.iloc[j]['low'] for j in range(max(0, i-2), i+1)]
            ranges_3d = [high_prices_3d[k] - low_prices_3d[k] for k in range(len(high_prices_3d))]
            low_freq_momentum = (close_t - close_t_3) / (np.std(ranges_3d) + 1e-10)
        else:
            low_freq_momentum = 0
            
        momentum_frequency_ratio = high_freq_momentum / (abs(low_freq_momentum) + 1e-10)
        
        # 11. Volume-Momentum Phase Alignment
        volume_momentum_coherence = np.sign(close_t - close_t_1) * np.sign(volume_t - volume_t_1)
        momentum_phase_intensity = (abs(close_t - close_t_1) * abs(volume_t - volume_t_1)) / price_range
        phase_alignment_score = volume_momentum_coherence * momentum_phase_intensity
        
        # 12. Regime-Dependent Momentum Propagation
        fast_regime_momentum = momentum_frequency_ratio if info_coherence_ratio > 1.2 else 0
        slow_regime_momentum = phase_alignment_score if info_coherence_ratio < 0.8 else 0
        momentum_propagation_factor = fast_regime_momentum * slow_regime_momentum
        
        # 13. Multi-Scale Volatility Patterns
        micro_volatility = price_range / (close_t + 1e-10)
        
        # Meso-scale volatility
        if i >= 2:
            high_prices_3d = [df.iloc[j]['high'] for j in range(max(0, i-2), i+1)]
            low_prices_3d = [df.iloc[j]['low'] for j in range(max(0, i-2), i+1)]
            ranges_3d = [high_prices_3d[k] - low_prices_3d[k] for k in range(len(high_prices_3d))]
            meso_volatility = np.std(ranges_3d) / (np.mean(ranges_3d) + 1e-10)
        else:
            meso_volatility = 1.0
            
        volatility_scale_ratio = micro_volatility / (meso_volatility + 1e-10)
        
        # 14. Volume Fractal Dynamics
        if i >= 3:
            volumes_3d = [df.iloc[j]['volume'] for j in range(max(0, i-2), i+1)]
            volume_scaling_exponent = np.std(volumes_3d) / (np.mean(volumes_3d) + 1e-10)
        else:
            volume_scaling_exponent = 1.0
            
        microstructure_fractal_dim = volatility_scale_ratio * volume_scaling_exponent
        
        # 15. Complexity-Regime Transitions
        high_complexity_regime = microstructure_fractal_dim if volatility_scale_ratio > 1.5 else 0
        low_complexity_regime = 0  # Simplified - would require correlation calculation
        complexity_transition_momentum = high_complexity_regime - low_complexity_regime
        
        # Integrated Alpha Synthesis
        flow_momentum_coupling = flow_imbalance * momentum_propagation_factor
        impact_regime_alignment = impact_regime_factor * info_regime_transition
        fractal_flow_efficiency = absorption_asymmetry * complexity_transition_momentum
        
        multi_frequency_alpha = momentum_frequency_ratio * microstructure_fractal_dim
        regime_adaptive_fusion = info_coherence_ratio * phase_alignment_score
        scale_invariant_signals = volatility_scale_ratio * liquidity_asymmetry
        
        high_freq_alpha = flow_momentum_coupling * multi_frequency_alpha
        medium_freq_alpha = impact_regime_alignment * regime_adaptive_fusion
        low_freq_alpha = fractal_flow_efficiency * scale_invariant_signals
        
        # Dynamic regime detection
        if info_coherence_ratio > 1.2:
            final_alpha = high_freq_alpha
        elif info_coherence_ratio < 0.8:
            final_alpha = low_freq_alpha
        else:
            final_alpha = medium_freq_alpha
            
        result.iloc[i] = final_alpha
    
    return result
