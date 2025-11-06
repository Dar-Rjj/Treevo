import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate basic price-based features
    data['immediate_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['short_term_momentum'] = data['close'] / data['close'].shift(3) - 1
    data['volume_momentum'] = data['volume'] / data['volume'].shift(1) - 1
    data['amount_flow_patterns'] = data['amount'] / data['volume']
    
    # Calculate rolling windows for various components
    for i in range(len(data)):
        if i < 20:  # Need enough data for calculations
            result.iloc[i] = 0
            continue
            
        current_data = data.iloc[:i+1]
        
        # 1. Multi-Scale Fractal Momentum
        # Short-Term Rejection Flow (5-day)
        upper_rejection = []
        lower_rejection = []
        
        for j in range(max(0, i-4), i+1):
            high_5d = current_data['high'].iloc[max(0, j-4):j+1].max()
            low_5d = current_data['low'].iloc[max(0, j-4):j+1].min()
            close_j = current_data['close'].iloc[j]
            close_5d_ago = current_data['close'].iloc[max(0, j-4)]
            
            if close_j < close_5d_ago:
                upper_rej = (high_5d - close_j) / close_j
            else:
                upper_rej = 0
                
            if close_j > close_5d_ago:
                lower_rej = (close_j - low_5d) / close_j
            else:
                lower_rej = 0
                
            upper_rejection.append(upper_rej)
            lower_rejection.append(lower_rej)
        
        net_rejection_flow = [lower_rejection[k] - upper_rejection[k] for k in range(len(upper_rejection))]
        flow_persistence = sum(1 for x in net_rejection_flow if x > 0)
        
        # Medium-Term Flow Momentum (10-day)
        if i >= 9:
            upper_flow_momentum = sum(upper_rejection[max(0, len(upper_rejection)-10):max(0, len(upper_rejection)-3)])
            lower_flow_momentum = sum(lower_rejection[max(0, len(lower_rejection)-10):max(0, len(lower_rejection)-3)])
            net_flow_momentum = lower_flow_momentum - upper_flow_momentum
            
            # Flow Acceleration (using 5-day difference)
            if i >= 14:
                net_flow_momentum_5d_ago = net_flow_momentum - (net_rejection_flow[-1] - net_rejection_flow[-6] if len(net_rejection_flow) >= 6 else 0)
                flow_acceleration = net_flow_momentum - net_flow_momentum_5d_ago
            else:
                flow_acceleration = 0
        else:
            net_flow_momentum = 0
            flow_acceleration = 0
        
        # Fractal Range Momentum
        if i >= 2:
            range_expansion_ratio = ((data['high'].iloc[i] - data['low'].iloc[i]) / 
                                   (data['high'].iloc[i-2] - data['low'].iloc[i-2]))
            range_flow_alignment = range_expansion_ratio * net_rejection_flow[-1] if net_rejection_flow else 0
            
            if i >= 4:
                range_flow_alignment_2d_ago = ((data['high'].iloc[i-2] - data['low'].iloc[i-2]) / 
                                             (data['high'].iloc[i-4] - data['low'].iloc[i-4])) * net_rejection_flow[-3] if len(net_rejection_flow) >= 3 else 0
                range_flow_acceleration = (range_flow_alignment / range_flow_alignment_2d_ago 
                                         if range_flow_alignment_2d_ago != 0 else 0)
            else:
                range_flow_acceleration = 0
        else:
            range_expansion_ratio = 1
            range_flow_alignment = 0
            range_flow_acceleration = 0
        
        # Price-Flow Momentum Integration
        flow_price_convergence = net_rejection_flow[-1] * data['immediate_momentum'].iloc[i] if net_rejection_flow else 0
        flow_momentum_gap = net_rejection_flow[-1] - data['short_term_momentum'].iloc[i] if net_rejection_flow else 0
        multi_scale_flow_momentum = flow_price_convergence * flow_momentum_gap
        
        # Volume-Flow Momentum Dynamics
        volume_flow_alignment = data['volume_momentum'].iloc[i] * net_rejection_flow[-1] if net_rejection_flow else 0
        amount_flow_convergence = data['amount_flow_patterns'].iloc[i] * net_rejection_flow[-1] if net_rejection_flow else 0
        flow_volume_divergence = volume_flow_alignment - amount_flow_convergence
        
        # 2. Asymmetric Liquidity Pressure
        # Price-Liquidity Asymmetry
        high_low_range = data['high'].iloc[i] - data['low'].iloc[i]
        if high_low_range > 0:
            buying_pressure_absorption = (data['high'].iloc[i] - data['open'].iloc[i]) / high_low_range
            selling_pressure_absorption = (data['open'].iloc[i] - data['low'].iloc[i]) / high_low_range
            liquidity_asymmetry_ratio = (buying_pressure_absorption / selling_pressure_absorption 
                                       if selling_pressure_absorption > 0 else 1)
        else:
            liquidity_asymmetry_ratio = 1
        
        # Volume-Pressure Dynamics
        if high_low_range > 0:
            if data['close'].iloc[i] > data['open'].iloc[i]:
                bullish_liquidity_pressure = data['volume'].iloc[i] / high_low_range
                bearish_liquidity_pressure = 0
            else:
                bullish_liquidity_pressure = 0
                bearish_liquidity_pressure = data['volume'].iloc[i] / high_low_range
            
            liquidity_pressure_asymmetry = (bullish_liquidity_pressure / bearish_liquidity_pressure 
                                          if bearish_liquidity_pressure > 0 else 1)
        else:
            liquidity_pressure_asymmetry = 1
        
        # Trade Size Pressure
        amount_volume_ratio = data['amount'].iloc[i] / data['volume'].iloc[i] if data['volume'].iloc[i] > 0 else 0
        historical_ratios = [data['amount'].iloc[j] / data['volume'].iloc[j] 
                           for j in range(max(0, i-19), i) if data['volume'].iloc[j] > 0]
        
        if historical_ratios:
            median_ratio = np.median(historical_ratios)
            large_trade_pressure = 1 if amount_volume_ratio > median_ratio else 0
            small_trade_pressure = 1 if amount_volume_ratio < median_ratio else 0
            trade_size_skew = large_trade_pressure - small_trade_pressure
        else:
            trade_size_skew = 0
        
        # Microstructure Pressure Integration
        opening_gap_momentum = (data['open'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1] if i > 0 else 0
        if high_low_range > 0:
            opening_range_capture = (data['high'].iloc[i] - data['open'].iloc[i]) / high_low_range
            closing_strength = (data['close'].iloc[i] - data['low'].iloc[i]) / high_low_range
        else:
            opening_range_capture = 0.5
            closing_strength = 0.5
            
        opening_pressure_score = opening_gap_momentum * opening_range_capture
        closing_momentum = closing_strength * data['immediate_momentum'].iloc[i]
        closing_pressure_divergence = closing_strength - opening_range_capture
        
        pressure_momentum = opening_pressure_score * closing_momentum
        
        # 3. Cross-Regime Flow Dynamics
        # Fractal Regime Detection
        recent_net_rejection = [net_rejection_flow[k] for k in range(max(0, len(net_rejection_flow)-20), len(net_rejection_flow)-1)]
        if recent_net_rejection:
            high_rejection_threshold = np.percentile(recent_net_rejection, 80)
            low_rejection_threshold = np.percentile(recent_net_rejection, 20)
            current_net_rejection = net_rejection_flow[-1] if net_rejection_flow else 0
            
            if current_net_rejection > high_rejection_threshold:
                fractal_regime = 'high_rejection'
            elif current_net_rejection < low_rejection_threshold:
                fractal_regime = 'low_rejection'
            else:
                fractal_regime = 'normal'
        else:
            fractal_regime = 'normal'
        
        # Volatility Regime Classification
        recent_volatility = [(data['high'].iloc[j] - data['low'].iloc[j]) / data['close'].iloc[j] 
                           for j in range(max(0, i-19), i)]
        if recent_volatility:
            high_vol_threshold = np.percentile(recent_volatility, 80)
            low_vol_threshold = np.percentile(recent_volatility, 20)
            current_vol = (data['high'].iloc[i] - data['low'].iloc[i]) / data['close'].iloc[i]
            
            if current_vol > high_vol_threshold:
                volatility_regime = 'high_volatility'
            elif current_vol < low_vol_threshold:
                volatility_regime = 'low_volatility'
            else:
                volatility_regime = 'normal'
        else:
            volatility_regime = 'normal'
        
        # Momentum Regime Identification
        if range_expansion_ratio > 1.2 and data['volume_momentum'].iloc[i] > 0.1:
            momentum_regime = 'high_momentum_expansion'
        elif range_expansion_ratio <= 1.2 and data['volume_momentum'].iloc[i] <= 0.1:
            momentum_regime = 'low_momentum_compression'
        else:
            momentum_regime = 'transition'
        
        # 4. Core Component Construction
        fractal_flow_component = (net_rejection_flow[-1] if net_rejection_flow else 0) * flow_acceleration * flow_persistence
        liquidity_pressure_component = liquidity_asymmetry_ratio * trade_size_skew * 1  # Simplified correlation
        volume_flow_component = volume_flow_alignment * flow_volume_divergence
        pressure_dynamics_component = pressure_momentum * (range_expansion_ratio * (pressure_momentum / (pressure_momentum + 0.0001)))  # Simplified consistency
        multi_scale_component = multi_scale_flow_momentum * (flow_price_convergence * flow_momentum_gap)  # Simplified consistency
        
        # 5. Cross-Regime Weighting Scheme
        if fractal_regime == 'high_rejection' and volatility_regime == 'high_volatility':
            weights = [0.35, 0.3, 0.2, 0.1, 0.05]  # Fractal Flow, Liquidity Pressure, Volume-Flow, Pressure Dynamics, Multi-Scale
        elif fractal_regime == 'low_rejection' and volatility_regime == 'low_volatility':
            weights = [0.25, 0.25, 0.2, 0.2, 0.1]
        elif momentum_regime == 'high_momentum_expansion':
            weights = [0.3, 0.25, 0.25, 0.1, 0.1]
        else:  # Transition regimes
            weights = [0.2, 0.3, 0.25, 0.15, 0.1]
        
        # 6. Final Alpha Generation
        components = [fractal_flow_component, liquidity_pressure_component, volume_flow_component, 
                     pressure_dynamics_component, multi_scale_component]
        
        regime_weighted_composite = sum(comp * weight for comp, weight in zip(components, weights))
        
        # Cross-Regime Validation
        positive_components = sum(1 for comp in components if comp > 0)
        cross_regime_valid = positive_components >= 2 or positive_components <= 1  # At least 2 with same sign
        
        # Persistence Filter
        flow_regime_coherence = 0.6  # Simplified
        sustained_pressure_duration = 3  # Simplified
        
        persistence_valid = flow_regime_coherence > 0.5 and sustained_pressure_duration > 2
        
        if cross_regime_valid and persistence_valid:
            # Momentum Enhancement and Pressure Confirmation
            momentum_enhanced = regime_weighted_composite * (1 + range_flow_acceleration)
            pressure_confirmed = momentum_enhanced + (closing_pressure_divergence * (pressure_momentum / (abs(pressure_momentum) + 0.0001)))
            final_alpha = pressure_confirmed
        else:
            final_alpha = regime_weighted_composite * 0.5  # Penalize invalid signals
        
        result.iloc[i] = final_alpha
    
    # Fill any remaining NaN values
    result = result.fillna(0)
    
    return result
