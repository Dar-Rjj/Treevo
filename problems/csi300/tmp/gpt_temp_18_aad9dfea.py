import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic price and volume metrics
    df['returns'] = df['close'].pct_change()
    df['volatility_daily'] = df['high'] - df['low']
    df['volume_change'] = df['volume'].pct_change()
    
    # Multi-Timeframe Synchronization Framework
    for i in range(len(df)):
        if i < 13:  # Need enough data for calculations
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]
        
        # Volatility-Volume Correlation Structure
        short_term_corr = current_data['close'].iloc[-3:].corr(current_data['volume'].iloc[-3:])
        medium_term_corr = current_data['close'].iloc[-6:].corr(current_data['volume'].iloc[-6:])
        long_term_corr = current_data['close'].iloc[-14:].corr(current_data['volume'].iloc[-14:])
        
        # Synchronization Strength Assessment
        vol_change = abs(current_data['volatility_daily'].iloc[-1] - current_data['volatility_daily'].iloc[-2])
        vol_change_abs = abs(vol_change)
        vol_change_pct = vol_change / (current_data['volatility_daily'].iloc[-2] + 1e-8)
        
        sync_magnitude = vol_change_abs * abs(current_data['volume_change'].iloc[-1])
        
        # Volatility Regime Confirmation
        volatility_expansion = current_data['volatility_daily'].iloc[-1] > current_data['volatility_daily'].iloc[-2]
        vol_past_5 = current_data['volatility_daily'].iloc[-6:-1]
        volatility_persistence = sum(vol_past_5.iloc[j] > vol_past_5.iloc[j-1] if j > 0 else False for j in range(len(vol_past_5)))
        volatility_momentum = (current_data['volatility_daily'].iloc[-1] / (current_data['volatility_daily'].iloc[-2] + 1e-8)) - 1
        
        # Bidirectional Flow Imbalance Integration
        high_low_range = current_data['high'].iloc[-1] - current_data['low'].iloc[-1]
        buy_pressure = current_data['volume'].iloc[-1] * (current_data['close'].iloc[-1] - current_data['low'].iloc[-1]) / (high_low_range + 1e-8)
        sell_pressure = current_data['volume'].iloc[-1] * (current_data['high'].iloc[-1] - current_data['close'].iloc[-1]) / (high_low_range + 1e-8)
        
        # Net pressure accumulation over 5 days
        net_pressure_5d = 0
        for j in range(min(5, i+1)):
            idx = i - j
            if idx >= 0:
                range_j = current_data['high'].iloc[idx] - current_data['low'].iloc[idx]
                bp_j = current_data['volume'].iloc[idx] * (current_data['close'].iloc[idx] - current_data['low'].iloc[idx]) / (range_j + 1e-8)
                sp_j = current_data['volume'].iloc[idx] * (current_data['high'].iloc[idx] - current_data['close'].iloc[idx]) / (range_j + 1e-8)
                net_pressure_5d += (bp_j - sp_j)
        
        # Price-Volume Convergence Patterns
        price_change_sign = np.sign(current_data['close'].iloc[-1] - current_data['close'].iloc[-2])
        volume_change_sign = np.sign(current_data['volume'].iloc[-1] - current_data['volume'].iloc[-2])
        volume_confirmation = price_change_sign * volume_change_sign
        
        divergence_magnitude = abs(current_data['close'].iloc[-1] - current_data['close'].iloc[-2]) / (abs(current_data['volume'].iloc[-1] - current_data['volume'].iloc[-2]) + 1)
        
        # Convergence persistence
        convergence_count = 0
        for j in range(min(5, i+1)):
            idx = i - j
            if idx > 0:
                pc_sign = np.sign(current_data['close'].iloc[idx] - current_data['close'].iloc[idx-1])
                vc_sign = np.sign(current_data['volume'].iloc[idx] - current_data['volume'].iloc[idx-1])
                if pc_sign * vc_sign > 0:
                    convergence_count += 1
                else:
                    break
        
        # Microstructural Efficiency Metrics
        price_discovery_efficiency = high_low_range / ((current_data['amount'].iloc[-1] / current_data['volume'].iloc[-1]) + 1e-8)
        
        # Volume clustering efficiency (last 5 days)
        vol_last_5 = current_data['volume'].iloc[-5:]
        volume_clustering_efficiency = vol_last_5.std() / (vol_last_5.mean() + 1e-8)
        
        range_utilization = (current_data['close'].iloc[-1] - current_data['open'].iloc[-1]) / (high_low_range + 1e-8)
        
        # Structural Regime Detection & Classification
        returns_5d = current_data['returns'].iloc[-5:]
        returns_20d = current_data['returns'].iloc[-20:]
        regime_volatility_ratio = returns_5d.std() / (returns_20d.std() + 1e-8)
        
        # Volume-volatility alignment
        vol_vol_alignment = np.sign(current_data['volume'].iloc[-1] - current_data['volume'].iloc[-2]) * np.sign(current_data['volatility_daily'].iloc[-1] - current_data['volatility_daily'].iloc[-2])
        
        # Liquidity Depth Assessment
        effective_spread_depth = high_low_range * current_data['volume'].iloc[-1] / (current_data['amount'].iloc[-1] + 1e-8)
        
        # Volume concentration
        vol_max_prev_4 = max(current_data['volume'].iloc[-5:-1]) if i >= 4 else current_data['volume'].iloc[-1]
        volume_concentration = current_data['volume'].iloc[-1] / (vol_max_prev_4 + 1e-8)
        
        depth_resilience = 1 - (abs(current_data['close'].iloc[-1] - current_data['close'].iloc[-2]) / (effective_spread_depth + 1e-8))
        
        # Structural Break Detection
        ma_volume_10 = current_data['volume'].iloc[-10:].mean()
        ma_volume_50 = current_data['volume'].iloc[-20:].mean() if i >= 19 else ma_volume_10
        volume_break_signals = (current_data['volume'].iloc[-1] > 2 * ma_volume_10) and (current_data['volume'].iloc[-1] > 1.5 * ma_volume_50)
        
        price_break_confirmation = abs(current_data['close'].iloc[-1] - current_data['close'].iloc[-2]) > 2 * returns_20d.std()
        
        # Cross-Dimensional Synchronization Dynamics
        # Sync multiplier based on efficiency regimes
        high_sync_regime = (abs(range_utilization) > 0.6) and (price_discovery_efficiency > 0.7)
        low_sync_regime = (abs(range_utilization) < 0.4) or (price_discovery_efficiency < 0.5)
        sync_multiplier = 1 + range_utilization * price_discovery_efficiency
        
        # Regime-Adaptive Signal Generation
        # Core synchronization generation
        core_sync = sync_magnitude * net_pressure_5d
        volume_confirmed_sync = core_sync * volume_confirmation
        efficiency_enhanced_sync = volume_confirmed_sync * sync_multiplier
        
        # Structural Break Exploitation
        break_momentum = 1 if volume_break_signals and price_break_confirmation else 0
        regime_transition = break_momentum * regime_volatility_ratio
        
        # Flow-Consistent Synchronization
        flow_momentum = net_pressure_5d * (1 - abs(volume_clustering_efficiency))
        depth_adjusted_sync = flow_momentum * depth_resilience
        multi_timeframe_validation = depth_adjusted_sync * convergence_count
        
        # Final Alpha Construction
        # Regime-Aware Signal Integration
        efficiency_regime_signals = efficiency_enhanced_sync * price_discovery_efficiency
        break_regime_signals = regime_transition * (1 / (volume_clustering_efficiency + 1e-8))
        flow_regime_signals = multi_timeframe_validation * (current_data['volume'].iloc[-1] / (current_data['volume'].iloc[-2] + 1e-8))
        
        # Combine regime signals
        final_signal = (efficiency_regime_signals + break_regime_signals + flow_regime_signals) / 3
        
        # Risk-Structural Refinement
        volatility_weighted = final_signal * (volatility_persistence / 5)
        liquidity_refined = volatility_weighted * effective_spread_depth
        
        # Use correlation structure for final adjustment
        avg_correlation = (short_term_corr + medium_term_corr + long_term_corr) / 3
        risk_adjusted_signal = liquidity_refined / (abs(avg_correlation) + 1e-8)
        
        result.iloc[i] = risk_adjusted_signal
    
    # Fill initial NaN values
    result = result.fillna(0)
    
    return result
