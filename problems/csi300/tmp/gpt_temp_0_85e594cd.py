import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Price-Volume Divergence with Efficiency Persistence Alpha
    """
    data = df.copy()
    
    # Helper function for True Range calculation
    def true_range(high, low, close_prev):
        return np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))
    
    # Initialize result series
    alpha = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i < 10:  # Need sufficient history
            alpha.iloc[i] = 0
            continue
            
        current_data = data.iloc[:i+1]
        
        # Multi-Timeframe Price-Volume Divergence Analysis
        # Volume-weighted price momentum calculation
        close_t = current_data['close'].iloc[i]
        
        # Short-term VWAP divergence (2-day)
        if i >= 2:
            vwap_short = (current_data['close'].iloc[i-2:i+1] * current_data['volume'].iloc[i-2:i+1]).sum() / current_data['volume'].iloc[i-2:i+1].sum()
            vwap_div_short = (close_t - vwap_short) / vwap_short if vwap_short != 0 else 0
        else:
            vwap_div_short = 0
            
        # Medium-term VWAP divergence (6-day)
        if i >= 6:
            vwap_medium = (current_data['close'].iloc[i-6:i+1] * current_data['volume'].iloc[i-6:i+1]).sum() / current_data['volume'].iloc[i-6:i+1].sum()
            vwap_div_medium = (close_t - vwap_medium) / vwap_medium if vwap_medium != 0 else 0
        else:
            vwap_div_medium = 0
            
        # Long-term VWAP divergence (10-day)
        vwap_long = (current_data['close'].iloc[i-10:i+1] * current_data['volume'].iloc[i-10:i+1]).sum() / current_data['volume'].iloc[i-10:i+1].sum()
        vwap_div_long = (close_t - vwap_long) / vwap_long if vwap_long != 0 else 0
        
        # Volume acceleration vs price momentum divergence
        if i >= 2:
            vol_accel = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-2] - 1 if current_data['volume'].iloc[i-2] != 0 else 0
            price_momentum = current_data['close'].iloc[i] / current_data['close'].iloc[i-2] - 1 if current_data['close'].iloc[i-2] != 0 else 0
            divergence_strength = vol_accel - price_momentum
        else:
            divergence_strength = 0
            
        # Amount-driven divergence detection
        if i >= 5:
            avg_amount = current_data['amount'].iloc[i-4:i].mean()
            large_trade_conc = current_data['amount'].iloc[i] / avg_amount if avg_amount != 0 else 0
        else:
            large_trade_conc = 0
            
        price_impact_eff = abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-1]) / current_data['amount'].iloc[i] if current_data['amount'].iloc[i] != 0 else 0
        
        # Regime-Based Efficiency Pattern Recognition
        # Multi-scale efficiency calculation
        intraday_eff = (current_data['high'].iloc[i] - current_data['low'].iloc[i]) / abs(current_data['open'].iloc[i] - current_data['close'].iloc[i]) if abs(current_data['open'].iloc[i] - current_data['close'].iloc[i]) != 0 else 0
        
        overnight_eff = abs(current_data['open'].iloc[i] - current_data['close'].iloc[i-1]) / true_range(current_data['high'].iloc[i], current_data['low'].iloc[i], current_data['close'].iloc[i-1]) if true_range(current_data['high'].iloc[i], current_data['low'].iloc[i], current_data['close'].iloc[i-1]) != 0 else 0
        
        if i >= 3:
            swing_range = sum(true_range(current_data['high'].iloc[j], current_data['low'].iloc[j], current_data['close'].iloc[j-1]) for j in range(i-2, i+1))
            swing_eff = abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-3]) / swing_range if swing_range != 0 else 0
        else:
            swing_eff = 0
            
        # Efficiency regime classification
        high_eff = intraday_eff > 2.0 and swing_eff > 0.8
        low_eff = intraday_eff < 1.2 and swing_eff < 0.4
        mixed_eff = not high_eff and not low_eff
        
        # Efficiency persistence analysis
        eff_streak = 1
        current_regime = 'high' if high_eff else 'low' if low_eff else 'mixed'
        
        for j in range(i-1, max(i-6, -1), -1):
            if j < 3:
                break
            prev_intraday_eff = (current_data['high'].iloc[j] - current_data['low'].iloc[j]) / abs(current_data['open'].iloc[j] - current_data['close'].iloc[j]) if abs(current_data['open'].iloc[j] - current_data['close'].iloc[j]) != 0 else 0
            prev_swing_eff = abs(current_data['close'].iloc[j] - current_data['close'].iloc[j-3]) / sum(true_range(current_data['high'].iloc[k], current_data['low'].iloc[k], current_data['close'].iloc[k-1]) for k in range(j-2, j+1)) if j >= 3 else 0
            
            prev_high_eff = prev_intraday_eff > 2.0 and prev_swing_eff > 0.8
            prev_low_eff = prev_intraday_eff < 1.2 and prev_swing_eff < 0.4
            prev_regime = 'high' if prev_high_eff else 'low' if prev_low_eff else 'mixed'
            
            if prev_regime == current_regime:
                eff_streak += 1
            else:
                break
        
        # Volume-Price Convergence Dynamics
        # Volume clustering analysis
        vol_median = current_data['volume'].iloc[i-9:i].median()
        vol_burst = current_data['volume'].iloc[i] > 2 * vol_median if not pd.isna(vol_median) else False
        
        sustained_vol_count = 0
        for j in range(max(i-4, 0), i+1):
            if current_data['volume'].iloc[j] > 1.5 * vol_median:
                sustained_vol_count += 1
        
        # Price-volume synchronization
        price_dir = np.sign(current_data['close'].iloc[i] - current_data['close'].iloc[i-1]) if i >= 1 else 0
        vol_dir = np.sign(current_data['volume'].iloc[i] - current_data['volume'].iloc[i-1]) if i >= 1 else 0
        direction_align = price_dir * vol_dir
        
        magnitude_corr = abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-1]) * current_data['volume'].iloc[i] if i >= 1 else 0
        
        # Regime-Adaptive Signal Construction
        # Divergence efficiency integration
        vol_weighted_div = (vwap_div_short * 0.4 + vwap_div_medium * 0.35 + vwap_div_long * 0.25)
        
        # Efficiency regime multiplier
        if high_eff:
            regime_multiplier = 1.2
        elif low_eff:
            regime_multiplier = 0.8
        else:
            regime_multiplier = 1.0
            
        # Persistence-based signal enhancement
        streak_weight = min(eff_streak / 5.0, 2.0)  # Cap at 2.0
        
        # Dynamic Threshold Adaptation
        # Volatility-adjusted thresholds
        price_volatility = current_data['close'].iloc[i-9:i+1].std()
        vol_volatility = current_data['volume'].iloc[i-9:i+1].std()
        
        vol_adj_factor = price_volatility / vol_volatility if vol_volatility != 0 else 1.0
        
        # Final Alpha Generation
        # Multi-dimensional signal integration
        price_volume_div_score = (vol_weighted_div * 0.6 + divergence_strength * 0.4) * regime_multiplier
        
        efficiency_persistence_rating = (intraday_eff * 0.3 + swing_eff * 0.4 + overnight_eff * 0.3) * streak_weight
        
        convergence_timing_indicator = (direction_align * 0.5 + (1 if vol_burst else -0.5) * 0.3 + (sustained_vol_count / 5.0) * 0.2)
        
        # Raw divergence efficiency value
        raw_div_eff = price_volume_div_score * efficiency_persistence_rating * vol_adj_factor
        
        # Regime-weighted signal strength
        regime_weighted_signal = raw_div_eff * (1 + 0.1 * convergence_timing_indicator)
        
        # Final alpha with dynamic context
        final_alpha = regime_weighted_signal * (1 - price_impact_eff * 10)  # Penalize high price impact
        
        alpha.iloc[i] = final_alpha
    
    return alpha
