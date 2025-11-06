import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(3, len(df)):
        if i < 4:  # Need at least 4 days of data for some calculations
            result.iloc[i] = 0
            continue
            
        # Current and past data
        current = df.iloc[i]
        prev1 = df.iloc[i-1]
        prev2 = df.iloc[i-2]
        prev3 = df.iloc[i-3]
        
        # Regime-Dependent Price Microstructure
        intraday_regime_momentum = ((current['close'] - current['open']) / 
                                   (current['high'] - current['low'] + 1e-8) * 
                                   np.sign(current['close'] - prev1['close']))
        
        # Multi-day Regime Persistence
        sign_changes = 0
        for j in range(i-4, i+1):
            if j >= 1 and j < len(df):
                current_sign = np.sign(df.iloc[j]['close'] - df.iloc[j-1]['close'])
                prev_sign = np.sign(df.iloc[j-1]['close'] - df.iloc[j-2]['close']) if j >= 2 else 0
                if current_sign == prev_sign and prev_sign != 0:
                    sign_changes += 1
        multi_day_persistence = sign_changes / 5.0
        
        # Regime Volatility Compression
        regime_vol_compression = ((current['high'] - current['low']) / 
                                 (prev1['high'] - prev1['low'] + 1e-8) * 
                                 np.sign(current['close'] - prev1['close']))
        
        # Regime Gap Dynamics
        gap_open = current['open'] - prev1['close']
        intraday_move = current['close'] - current['open']
        regime_gap_dynamics = (np.sign(gap_open) if abs(gap_open) > 1e-8 else 0) * \
                             (np.sign(intraday_move) if abs(intraday_move) > 1e-8 else 0)
        
        # Volume-Microstructure Regime Detection
        volume_regime_confirmation = (np.sign(current['volume'] - prev1['volume']) * 
                                    np.sign(current['close'] - prev1['close']))
        
        multi_day_volume_regime = ((current['volume'] / (prev3['volume'] + 1e-8) - 1) * 
                                  (current['close'] / (prev3['close'] + 1e-8) - 1) * 
                                  np.sign(current['volume'] - prev3['volume']))
        
        volume_weighted_microstructure = (abs(current['close'] - prev1['close']) * 
                                        current['volume'] / (current['high'] - current['low'] + 1e-8))
        
        volume_regime_divergence = volume_regime_confirmation - multi_day_volume_regime * intraday_regime_momentum
        
        # Efficiency-Based Regime Classification
        intraday_efficiency_signal = (abs(current['close'] - current['open']) / 
                                    (current['high'] - current['low'] + 1e-8) * 
                                    np.sign(current['close'] - prev1['close']))
        
        # Multi-day price range
        high_range = max(df.iloc[i-2]['high'], df.iloc[i-1]['high'], current['high'])
        low_range = min(df.iloc[i-2]['low'], df.iloc[i-1]['low'], current['low'])
        multi_day_efficiency_regime = (abs(current['close'] - prev3['close']) / 
                                     (high_range - low_range + 1e-8) * 
                                     np.sign(current['close'] - prev3['close']))
        
        efficiency_volume_integration = (intraday_efficiency_signal * 
                                       (current['volume'] / (prev1['volume'] + 1e-8)) * 
                                       np.sign(current['close'] - current['open']))
        
        efficiency_regime_divergence = intraday_efficiency_signal - multi_day_efficiency_regime * volume_regime_confirmation
        
        # Flow-Based Regime Momentum
        microstructure_flow = ((current['close'] - prev1['close']) * 
                              current['volume'] / (abs(current['close'] - prev1['close']) + 1e-8))
        
        multi_day_flow_regime = ((current['close'] - prev3['close']) * 
                                (current['volume'] - prev3['volume']) / 
                                (abs(current['close'] - prev3['close']) + 1e-8))
        
        # Flow Regime Persistence
        flow_persistence_count = 0
        for j in range(i-2, i+1):
            if j >= 1 and j < len(df):
                current_flow = ((df.iloc[j]['close'] - df.iloc[j-1]['close']) * 
                              df.iloc[j]['volume'] / (abs(df.iloc[j]['close'] - df.iloc[j-1]['close']) + 1e-8))
                prev_flow = ((df.iloc[j-1]['close'] - df.iloc[j-2]['close']) * 
                           df.iloc[j-1]['volume'] / (abs(df.iloc[j-1]['close'] - df.iloc[j-2]['close']) + 1e-8)) if j >= 2 else 0
                if current_flow * prev_flow > 0:
                    flow_persistence_count += 1
        flow_regime_persistence = flow_persistence_count / 3.0
        
        flow_regime_divergence = microstructure_flow - multi_day_flow_regime * intraday_regime_momentum
        
        # Regime Transition Detection
        price_regime_transition = (np.sign(current['close'] - prev1['close']) * 
                                 np.sign(prev1['close'] - prev2['close']) * 
                                 np.sign(prev2['close'] - prev3['close']))
        
        volume_regime_transition = (np.sign(current['volume'] - prev1['volume']) * 
                                  np.sign(prev1['volume'] - prev2['volume']) * 
                                  np.sign(prev2['volume'] - prev3['volume']))
        
        efficiency_regime_transition = (np.sign(intraday_efficiency_signal) * 
                                      np.sign(multi_day_efficiency_regime) * 
                                      volume_regime_confirmation)
        
        flow_regime_transition = (np.sign(microstructure_flow) * 
                                np.sign(multi_day_flow_regime) * 
                                flow_regime_persistence)
        
        # Multi-Scale Regime Integration
        price_volume_regime_alignment = (intraday_regime_momentum * 
                                       volume_regime_confirmation * 
                                       multi_day_persistence)
        
        efficiency_flow_regime_convergence = (intraday_efficiency_signal * 
                                            microstructure_flow * 
                                            efficiency_regime_divergence)
        
        gap_regime_integration = (regime_gap_dynamics * 
                                volume_weighted_microstructure * 
                                flow_regime_persistence)
        
        transition_enhanced_regime = (price_regime_transition * 
                                    volume_regime_transition * 
                                    efficiency_regime_transition)
        
        # Regime-Aware Alpha Components
        core_regime_momentum = price_volume_regime_alignment * flow_regime_persistence
        efficiency_regime_factor = efficiency_flow_regime_convergence * volume_regime_divergence
        gap_regime_signal = gap_regime_integration * regime_gap_dynamics
        transition_regime_factor = transition_enhanced_regime * flow_regime_transition
        
        # Dynamic Regime Weighting
        regime_weight = 1.0
        
        # High Momentum Regime
        if core_regime_momentum > 0 and volume_regime_confirmation > 0:
            regime_weight *= 1.5
        
        # Low Momentum Regime  
        if core_regime_momentum < 0 and volume_regime_confirmation < 0:
            regime_weight *= 0.7
        
        # Efficiency Boost
        if intraday_efficiency_signal > 0.5 and efficiency_regime_divergence > 0:
            regime_weight *= 1.3
        
        # Transition Multiplier
        if transition_enhanced_regime > 0 and flow_regime_transition > 0:
            regime_weight *= 1.4
        
        # Validated Regime Signals
        confirmed_regime_momentum = core_regime_momentum * price_volume_regime_alignment
        robust_efficiency_regime = efficiency_regime_factor * efficiency_flow_regime_convergence
        validated_gap_regime = gap_regime_signal * gap_regime_integration
        confirmed_transition = transition_regime_factor * transition_enhanced_regime
        
        # Final Composite Alpha
        primary_factor = confirmed_regime_momentum * multi_day_persistence
        secondary_factor = robust_efficiency_regime * volume_regime_divergence
        tertiary_factor = validated_gap_regime * regime_gap_dynamics
        quaternary_factor = confirmed_transition * flow_regime_transition
        
        # Composite Regime Alpha with dynamic weighting
        composite_alpha = (primary_factor + secondary_factor + tertiary_factor + quaternary_factor) * regime_weight
        
        result.iloc[i] = composite_alpha
    
    # Fill initial values
    result = result.fillna(0)
    
    return result
