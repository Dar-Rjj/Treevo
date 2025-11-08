import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum Transmission with Microstructure Convergence alpha factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):  # Start from day 20 to have enough history
        current_data = df.iloc[:i+1]  # Only use current and past data
        
        # Multi-Timeframe Momentum Dynamics
        # Short-Term Momentum (3-day)
        if i >= 6:
            mom_3d = current_data['close'].iloc[i] / current_data['close'].iloc[i-3] - 1
            mom_6d = current_data['close'].iloc[i] / current_data['close'].iloc[i-6] - 1
            mom_accel = mom_3d - mom_6d
            
            # Momentum persistence (consecutive positive momentum days)
            mom_persistence = 0
            for j in range(min(5, i-3)):  # Check up to 5 days back
                if current_data['close'].iloc[i-j] / current_data['close'].iloc[i-j-3] - 1 > 0:
                    mom_persistence += 1
                else:
                    break
        else:
            mom_3d = mom_accel = mom_persistence = 0
        
        # Medium-Term Momentum (15-day)
        if i >= 15:
            mom_15d = current_data['close'].iloc[i] / current_data['close'].iloc[i-15] - 1
            
            # Momentum stability (variance of 3-day returns)
            recent_returns = []
            for j in range(min(5, i-2)):  # Last 5 periods of 3-day returns
                if i-j-3 >= 0:
                    ret = current_data['close'].iloc[i-j] / current_data['close'].iloc[i-j-3] - 1
                    recent_returns.append(ret)
            mom_stability = 1 / (np.std(recent_returns) + 1e-8) if recent_returns else 0
            
            # Momentum trend strength (slope of 15-day momentum)
            if i >= 30:
                mom_trend = []
                for j in range(3):  # Last 3 periods of 15-day momentum
                    if i-j-15 >= 0:
                        mom_val = current_data['close'].iloc[i-j] / current_data['close'].iloc[i-j-15] - 1
                        mom_trend.append(mom_val)
                if len(mom_trend) >= 2:
                    x = np.arange(len(mom_trend))
                    slope = np.polyfit(x, mom_trend, 1)[0]
                    mom_trend_strength = slope * 10  # Scale for better sensitivity
                else:
                    mom_trend_strength = 0
            else:
                mom_trend_strength = 0
        else:
            mom_15d = mom_stability = mom_trend_strength = 0
        
        # Regime-Adaptive Momentum Classification
        if i >= 25:
            # Momentum regime based on 10-day momentum percentile
            recent_mom = []
            for j in range(20):  # Last 20 periods of 10-day momentum
                if i-j-10 >= 0:
                    mom_val = current_data['close'].iloc[i-j] / current_data['close'].iloc[i-j-10] - 1
                    recent_mom.append(mom_val)
            if recent_mom:
                current_10d_mom = current_data['close'].iloc[i] / current_data['close'].iloc[i-10] - 1
                mom_regime = np.percentile(recent_mom, 75) if current_10d_mom > np.median(recent_mom) else np.percentile(recent_mom, 25)
            else:
                mom_regime = 0
        else:
            mom_regime = 0
        
        # Volatility-Microstructure Convergence
        # Volatility Dynamics
        if i >= 3:
            # ATR calculation
            def calculate_atr(data, idx, period=14):
                if idx < period:
                    return 0
                tr_sum = 0
                for j in range(period):
                    high_low = data['high'].iloc[idx-j] - data['low'].iloc[idx-j]
                    high_close = abs(data['high'].iloc[idx-j] - data['close'].iloc[idx-j-1]) if idx-j-1 >= 0 else 0
                    low_close = abs(data['low'].iloc[idx-j] - data['close'].iloc[idx-j-1]) if idx-j-1 >= 0 else 0
                    tr = max(high_low, high_close, low_close)
                    tr_sum += tr
                return tr_sum / period
            
            atr_current = calculate_atr(current_data, i)
            atr_3d_ago = calculate_atr(current_data, i-3) if i-3 >= 14 else atr_current
            vol_momentum = (atr_current / (atr_3d_ago + 1e-8)) - 1 if atr_3d_ago > 0 else 0
            
            # Volatility efficiency
            true_range = max(current_data['high'].iloc[i] - current_data['low'].iloc[i],
                           abs(current_data['high'].iloc[i] - current_data['close'].iloc[i-1]) if i-1 >= 0 else 0,
                           abs(current_data['low'].iloc[i] - current_data['close'].iloc[i-1]) if i-1 >= 0 else 0)
            vol_efficiency = abs(current_data['close'].iloc[i] - current_data['open'].iloc[i]) / (true_range + 1e-8)
            
            # Volatility persistence
            vol_persistence = 0
            for j in range(min(3, i-1)):
                if i-j-1 >= 0:
                    current_vol_dir = 1 if current_data['high'].iloc[i-j] - current_data['low'].iloc[i-j] > \
                                        current_data['high'].iloc[i-j-1] - current_data['low'].iloc[i-j-1] else -1
                    prev_vol_dir = 1 if current_data['high'].iloc[i-j-1] - current_data['low'].iloc[i-j-1] > \
                                      current_data['high'].iloc[i-j-2] - current_data['low'].iloc[i-j-2] else -1
                    if current_vol_dir == prev_vol_dir:
                        vol_persistence += 1
                    else:
                        break
        else:
            vol_momentum = vol_efficiency = vol_persistence = 0
        
        # Microstructure-Volatility Alignment
        if i >= 10:
            # Order flow clustering approximation using volume patterns
            recent_volumes = []
            for j in range(10):
                if i-j >= 0:
                    recent_volumes.append(current_data['volume'].iloc[i-j])
            
            volume_clustering = np.std(recent_volumes) / (np.mean(recent_volumes) + 1e-8)
            
            # Order flow-volatility divergence
            of_vol_divergence = volume_clustering * vol_momentum
            
            # Momentum-microstructure convergence
            mom_micro_convergence = mom_3d * (1 / (volume_clustering + 1e-8))
            
            # Efficiency stability (variance of 3-day efficiency ratios)
            efficiency_ratios = []
            for j in range(min(3, i-2)):
                if i-j-1 >= 0:
                    tr = max(current_data['high'].iloc[i-j] - current_data['low'].iloc[i-j],
                           abs(current_data['high'].iloc[i-j] - current_data['close'].iloc[i-j-1]),
                           abs(current_data['low'].iloc[i-j] - current_data['close'].iloc[i-j-1]))
                    eff = abs(current_data['close'].iloc[i-j] - current_data['open'].iloc[i-j]) / (tr + 1e-8)
                    efficiency_ratios.append(eff)
            efficiency_stability = 1 / (np.std(efficiency_ratios) + 1e-8) if efficiency_ratios else 0
        else:
            of_vol_divergence = mom_micro_convergence = efficiency_stability = 0
        
        # Liquidity-Momentum Transmission Engine
        if i >= 6:
            # Liquidity momentum
            vol_3d = sum(current_data['volume'].iloc[i-2:i+1]) if i >= 2 else current_data['volume'].iloc[i]
            vol_6d = sum(current_data['volume'].iloc[i-5:i+1]) if i >= 5 else vol_3d
            amt_3d = sum(current_data['amount'].iloc[i-2:i+1]) if i >= 2 else current_data['amount'].iloc[i]
            amt_6d = sum(current_data['amount'].iloc[i-5:i+1]) if i >= 5 else amt_3d
            
            liq_momentum = (vol_3d/(vol_6d + 1e-8)) - (amt_3d/(amt_6d + 1e-8))
            
            # Liquidity acceleration (simplified)
            if i >= 7:
                prev_vol_3d = sum(current_data['volume'].iloc[i-3:i]) if i >= 3 else current_data['volume'].iloc[i-1]
                prev_vol_6d = sum(current_data['volume'].iloc[i-6:i]) if i >= 6 else prev_vol_3d
                prev_amt_3d = sum(current_data['amount'].iloc[i-3:i]) if i >= 3 else current_data['amount'].iloc[i-1]
                prev_amt_6d = sum(current_data['amount'].iloc[i-6:i]) if i >= 6 else prev_amt_3d
                
                prev_liq_momentum = (prev_vol_3d/(prev_vol_6d + 1e-8)) - (prev_amt_3d/(prev_amt_6d + 1e-8))
                liq_accel = liq_momentum - prev_liq_momentum
            else:
                liq_accel = 0
            
            # Liquidity persistence
            liq_persistence = 0
            for j in range(min(3, i-1)):
                if i-j-1 >= 0:
                    current_liq_dir = 1 if current_data['volume'].iloc[i-j] > current_data['volume'].iloc[i-j-1] else -1
                    prev_liq_dir = 1 if current_data['volume'].iloc[i-j-1] > current_data['volume'].iloc[i-j-2] else -1
                    if current_liq_dir == prev_liq_dir:
                        liq_persistence += 1
                    else:
                        break
        else:
            liq_momentum = liq_accel = liq_persistence = 0
        
        # Momentum-Liquidity Interaction
        liq_mom_alignment = liq_momentum - mom_3d if i >= 6 else 0
        liq_confirmation = liq_accel * mom_accel if i >= 7 else 0
        liq_efficiency = liq_momentum * vol_efficiency if i >= 3 else 0
        
        # Breakout Confirmation with Microstructure Convergence
        if i >= 4:
            # Price Breakout Detection
            recent_high = max(current_data['high'].iloc[i-4:i+1])
            relative_strength = 1 if current_data['close'].iloc[i] > recent_high else -1
            
            # Breakout persistence
            breakout_persistence = 0
            for j in range(min(3, i-3)):
                if i-j-3 >= 0:
                    prev_recent_high = max(current_data['high'].iloc[i-j-4:i-j+1])
                    if current_data['close'].iloc[i-j] > prev_recent_high:
                        breakout_persistence += 1
                    else:
                        break
            
            # Breakout magnitude
            breakout_magnitude = (current_data['high'].iloc[i] - current_data['close'].iloc[i-1]) / (current_data['close'].iloc[i-1] + 1e-8)
            
            # Liquidity surge detection
            if i >= 10:
                avg_money_flow = np.mean(current_data['amount'].iloc[i-9:i+1])
                current_money_flow = current_data['amount'].iloc[i]
                liquidity_surge = 1 if current_money_flow > 1.2 * avg_money_flow else 0
            else:
                liquidity_surge = 0
        else:
            relative_strength = breakout_persistence = breakout_magnitude = liquidity_surge = 0
        
        # Composite Convergence Alpha Construction
        # Core Convergence Component
        momentum_micro_score = (mom_3d * 0.3 + mom_15d * 0.2 + mom_micro_convergence * 0.5)
        volatility_convergence_quality = (efficiency_stability * 0.4 + (1 / (abs(vol_momentum) + 1e-8)) * 0.3 + vol_efficiency * 0.3)
        liquidity_transmission = (liq_momentum * 0.4 + liq_confirmation * 0.3 + liq_efficiency * 0.3)
        
        core_convergence = momentum_micro_score * volatility_convergence_quality * (1 + liquidity_transmission)
        
        # Divergence Enhancement
        volatility_micro_divergence = of_vol_divergence * 0.6 + mom_accel * 0.4
        multi_timeframe_divergence = (mom_3d - mom_15d) * 0.5 + mom_trend_strength * 0.5
        
        divergence_component = volatility_micro_divergence * multi_timeframe_divergence * mom_persistence
        
        # Breakout Confirmation Overlay
        breakout_strength = (relative_strength * 0.4 + breakout_persistence * 0.3 + breakout_magnitude * 0.3)
        liquidity_momentum_confirmation = liquidity_surge * 0.6 + liq_mom_alignment * 0.4
        
        breakout_overlay = breakout_strength * (1 + liquidity_momentum_confirmation) * core_convergence
        
        # Regime-Adaptive Signal Modulation
        volatility_regime_scale = 1 / (abs(vol_momentum) + 0.1)
        microstructure_regime_adjust = 1 / (volume_clustering + 0.1) if i >= 10 else 1
        transmission_amplification = 1 + (liq_persistence * 0.1 + vol_persistence * 0.1)
        
        regime_modulation = volatility_regime_scale * microstructure_regime_adjust * transmission_amplification
        
        # Final Convergence Alpha Signal
        final_alpha = (core_convergence * 0.5 + 
                      divergence_component * 0.3 + 
                      breakout_overlay * 0.2) * regime_modulation
        
        alpha.iloc[i] = final_alpha
    
    # Fill initial NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
