import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate required components with sufficient lookback
    for i in range(6, len(df)):
        current_data = df.iloc[:i+1]
        
        # Multi-Timeframe Momentum Efficiency
        # Short-term Momentum (3-day)
        if i >= 6:
            mom_3d = current_data['close'].iloc[i] / current_data['close'].iloc[i-3] - 1
            mom_3d_prev = current_data['close'].iloc[i-3] / current_data['close'].iloc[i-6] - 1
            mom_accel = mom_3d - mom_3d_prev
            mom_coherence = mom_3d * mom_3d_prev
            
            # Momentum Persistence
            mom_persistence = 0
            consecutive_dir = 0
            for j in range(1, min(6, i+1)):
                ret_j = current_data['close'].iloc[i-j+1] / current_data['close'].iloc[i-j] - 1
                if j == 1:
                    consecutive_dir = 1 if ret_j > 0 else -1
                    mom_persistence = ret_j
                else:
                    current_dir = 1 if ret_j > 0 else -1
                    if current_dir == consecutive_dir:
                        mom_persistence += ret_j
                    else:
                        break
        else:
            mom_3d = mom_accel = mom_coherence = mom_persistence = 0
        
        # Medium-term Momentum (5-day)
        if i >= 5:
            mom_5d = current_data['close'].iloc[i] / current_data['close'].iloc[i-5] - 1
        else:
            mom_5d = 0
        
        # Dynamic Volatility-Order Flow Integration
        # Range-Based Volatility with Microstructure
        daily_range = (current_data['high'].iloc[i] - current_data['low'].iloc[i]) / current_data['close'].iloc[i]
        
        # Volatility Concentration
        if i >= 4:
            range_sum = sum(current_data['high'].iloc[i-j] - current_data['low'].iloc[i-j] for j in range(5))
            vol_concentration = (current_data['high'].iloc[i] - current_data['low'].iloc[i]) / (range_sum / 5)
        else:
            vol_concentration = 1
        
        # Microstructure Order Intensity
        if current_data['amount'].iloc[i] > 0:
            microstructure_intensity = (current_data['high'].iloc[i] - current_data['low'].iloc[i]) * current_data['volume'].iloc[i] / current_data['amount'].iloc[i]
        else:
            microstructure_intensity = 0
        
        # Asymmetric Volatility Dynamics
        if i >= 1:
            if current_data['close'].iloc[i] > current_data['close'].iloc[i-1]:
                up_vol = (current_data['high'].iloc[i] - current_data['open'].iloc[i]) / current_data['close'].iloc[i-1]
                down_vol = 0
            elif current_data['close'].iloc[i] < current_data['close'].iloc[i-1]:
                up_vol = 0
                down_vol = (current_data['open'].iloc[i] - current_data['low'].iloc[i]) / current_data['close'].iloc[i-1]
            else:
                up_vol = down_vol = 0
            
            if down_vol > 0:
                vol_asymmetry_ratio = up_vol / down_vol if up_vol > 0 else 1
            else:
                vol_asymmetry_ratio = 1
        else:
            up_vol = down_vol = vol_asymmetry_ratio = 0
        
        # Volume-Volatility Entanglement
        if i >= 6:
            vol_accel_3d = (current_data['volume'].iloc[i] / current_data['volume'].iloc[i-3] - 1) - (current_data['volume'].iloc[i-3] / current_data['volume'].iloc[i-6] - 1)
            vol_vol_alignment = vol_accel_3d * vol_concentration
        else:
            vol_accel_3d = vol_vol_alignment = 0
        
        # Pressure Efficiency
        if (current_data['high'].iloc[i] - current_data['low'].iloc[i]) > 0:
            pressure_efficiency = ((current_data['close'].iloc[i] - current_data['open'].iloc[i]) * current_data['volume'].iloc[i] - 
                                 (current_data['open'].iloc[i] - current_data['close'].iloc[i-1]) * current_data['volume'].iloc[i]) / (current_data['high'].iloc[i] - current_data['low'].iloc[i])
        else:
            pressure_efficiency = 0
        
        # Microstructure Anchoring Framework
        # Volume-Weighted Price Anchors
        if i >= 4:
            vwap_high = sum(current_data['high'].iloc[i-j] * current_data['volume'].iloc[i-j] for j in range(5)) / sum(current_data['volume'].iloc[i-j] for j in range(5))
            vwap_low = sum(current_data['low'].iloc[i-j] * current_data['volume'].iloc[i-j] for j in range(5)) / sum(current_data['volume'].iloc[i-j] for j in range(5))
            
            if (current_data['high'].iloc[i] - current_data['low'].iloc[i]) > 0:
                anchor_efficiency = (vwap_high - vwap_low) / (current_data['high'].iloc[i] - current_data['low'].iloc[i])
            else:
                anchor_efficiency = 1
        else:
            vwap_high = current_data['high'].iloc[i]
            vwap_low = current_data['low'].iloc[i]
            anchor_efficiency = 1
        
        # Order Flow Efficiency
        opening_order_eff = (current_data['open'].iloc[i] - current_data['close'].iloc[i-1]) * (current_data['amount'].iloc[i] / current_data['volume'].iloc[i]) if current_data['volume'].iloc[i] > 0 else 0
        
        if i >= 1 and current_data['amount'].iloc[i-1] > 0:
            closing_pressure = (current_data['close'].iloc[i] - (current_data['high'].iloc[i] + current_data['low'].iloc[i])/2) * (current_data['amount'].iloc[i] - current_data['amount'].iloc[i-1]) / current_data['amount'].iloc[i-1]
        else:
            closing_pressure = 0
        
        if (current_data['high'].iloc[i] - current_data['open'].iloc[i] + current_data['open'].iloc[i] - current_data['low'].iloc[i]) > 0:
            gap_efficiency = (current_data['close'].iloc[i] - current_data['open'].iloc[i]) / ((current_data['high'].iloc[i] - current_data['open'].iloc[i]) + (current_data['open'].iloc[i] - current_data['low'].iloc[i]))
        else:
            gap_efficiency = 0
        
        # Behavioral Anchoring Patterns
        high_breakout = (current_data['close'].iloc[i] - vwap_high) * current_data['volume'].iloc[i] if current_data['close'].iloc[i] > vwap_high else 0
        low_breakout = -(vwap_low - current_data['close'].iloc[i]) * current_data['volume'].iloc[i] if current_data['close'].iloc[i] < vwap_low else 0
        
        # Anchor Persistence
        anchor_persistence = 0
        if i >= 1:
            current_anchor_pos = 1 if current_data['close'].iloc[i] > (vwap_high + vwap_low)/2 else -1
            for j in range(1, min(5, i+1)):
                prev_anchor_pos = 1 if current_data['close'].iloc[i-j] > (vwap_high + vwap_low)/2 else -1
                if prev_anchor_pos == current_anchor_pos:
                    anchor_persistence += 1
                else:
                    break
        
        # Adaptive Regime-Dependent Signal Enhancement
        # Volatility-Adjusted Momentum
        vol_adj_mom = mom_5d / daily_range if daily_range > 0 else mom_5d
        mom_scaled = mom_3d * vol_asymmetry_ratio
        mom_efficiency = mom_persistence / (current_data['high'].iloc[i] - current_data['low'].iloc[i]) if (current_data['high'].iloc[i] - current_data['low'].iloc[i]) > 0 else 0
        
        # Volume-Enhanced Confirmation
        vol_mom_alignment = np.sign(mom_accel) * np.sign(vol_accel_3d) if (mom_accel != 0 and vol_accel_3d != 0) else 0
        vol_anchor_conf = vol_accel_3d * (high_breakout + low_breakout)
        order_flow_weight = opening_order_eff * closing_pressure
        
        # Dynamic Regime Weighting
        vol_regime = 1 + abs(vol_asymmetry_ratio)
        mom_regime = 1 + abs(mom_coherence)
        anchor_regime = 1 + abs(anchor_efficiency)
        order_flow_regime = 1 + abs(opening_order_eff - closing_pressure)
        
        # Composite Alpha Generation
        # Core Signal Integration
        vol_entangled_mom = vol_adj_mom * vol_vol_alignment
        microstructure_order_flow = order_flow_weight * microstructure_intensity
        anchor_confirmed_breakouts = (high_breakout + low_breakout) * (anchor_persistence + 1)
        
        # Adaptive Signal Enhancement
        regime_weighted_core = (vol_entangled_mom * vol_regime + 
                              microstructure_order_flow * order_flow_regime + 
                              anchor_confirmed_breakouts * anchor_regime)
        
        volume_confirmed_mom = regime_weighted_core * vol_mom_alignment if vol_mom_alignment != 0 else regime_weighted_core
        efficiency_enhanced = volume_confirmed_mom * gap_efficiency * pressure_efficiency
        
        # Final Alpha Factor
        multi_scale_base = efficiency_enhanced * mom_efficiency
        microstructure_entangled = multi_scale_base * anchor_efficiency * vol_anchor_conf
        
        result.iloc[i] = microstructure_entangled
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
