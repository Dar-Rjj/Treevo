import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(4, len(df)):
        if i < 4:
            result.iloc[i] = 0
            continue
            
        # Extract current and historical data
        current = df.iloc[i]
        prev1 = df.iloc[i-1]
        prev2 = df.iloc[i-2]
        prev3 = df.iloc[i-3]
        prev4 = df.iloc[i-4]
        
        # Avoid division by zero
        eps = 1e-8
        
        # Volume ratios
        vol_ratio = current['volume'] / (prev1['volume'] + eps)
        
        # 1. Multi-Regime Fractal Classification
        # Flow-Weighted Fractal Volatility
        if (prev1['high'] - prev1['low']) > eps:
            flow_weighted_vol = ((current['high'] - current['low']) / (prev1['high'] - prev1['low'])) * vol_ratio
        else:
            flow_weighted_vol = 0
            
        # Opening Absorption Efficiency
        if abs(prev1['open'] - prev1['close']) > eps:
            opening_absorption = (abs(current['close'] - current['open']) / abs(prev1['open'] - prev1['close'])) * vol_ratio
        else:
            opening_absorption = 0
            
        # Fractal Absorption Intensity
        if (current['high'] - current['low']) > eps and (prev1['high'] - prev1['low']) > eps:
            fractal_intensity = (current['volume'] / (current['high'] - current['low'])) * (prev1['volume'] / (prev1['high'] - prev1['low'])) * vol_ratio
        else:
            fractal_intensity = 0
            
        # Fractal Absorption Surge
        vol_ma_5 = (prev4['volume'] + prev3['volume'] + prev2['volume'] + prev1['volume'] + current['volume']) / 5
        if current['volume'] > 1.5 * vol_ma_5:
            fractal_surge = vol_ratio
        else:
            fractal_surge = 0
            
        # 2. Asymmetric Fractal Microstructure
        # Upside Fractal Absorption
        upside_absorption = (current['high'] - max(current['open'], current['close'])) * vol_ratio
        
        # Downside Fractal Resistance
        downside_resistance = (min(current['open'], current['close']) - current['low']) * vol_ratio
        
        # Net Fractal Asymmetry
        net_asymmetry = upside_absorption - downside_resistance
        
        # Fractal Intraday Efficiency
        if (current['high'] - current['low']) > eps:
            fractal_efficiency = (abs(current['close'] - current['open']) / (current['high'] - current['low'])) * vol_ratio
        else:
            fractal_efficiency = 0
            
        # Fractal Efficiency Momentum
        prev_efficiency = abs(prev1['close'] - prev1['open']) / (prev1['high'] - prev1['low'] + eps) * (prev1['volume'] / (df.iloc[i-2]['volume'] + eps))
        efficiency_momentum = (fractal_efficiency / (prev_efficiency + eps) - 1) * vol_ratio
        
        # Short-term High Absorption
        short_term_high = (current['high'] - max(prev2['close'], prev1['close'], current['close'])) * vol_ratio
        
        # Fractal Persistence
        asymmetry_signs = []
        for j in range(max(0, i-2), i+1):
            if j >= 2:
                high_j = df.iloc[j]['high']
                open_j = df.iloc[j]['open']
                close_j = df.iloc[j]['close']
                low_j = df.iloc[j]['low']
                vol_prev = df.iloc[j-1]['volume']
                vol_ratio_j = df.iloc[j]['volume'] / (vol_prev + eps)
                
                up_abs_j = (high_j - max(open_j, close_j)) * vol_ratio_j
                down_res_j = (min(open_j, close_j) - low_j) * vol_ratio_j
                net_asy_j = up_abs_j - down_res_j
                asymmetry_signs.append(np.sign(net_asy_j))
        
        if len(asymmetry_signs) >= 2:
            persistence_count = sum(1 for k in range(1, len(asymmetry_signs)) if asymmetry_signs[k] == asymmetry_signs[k-1])
            fractal_persistence = (persistence_count / (len(asymmetry_signs) - 1)) * vol_ratio
        else:
            fractal_persistence = 0
            
        # 3. Volume-Velocity Integration
        # Upside Absorption Volume
        if (current['high'] - current['open']) > eps:
            upside_abs_vol = (current['volume'] / (current['high'] - current['open'])) * vol_ratio
        else:
            upside_abs_vol = 0
            
        # Downside Resistance Volume
        if (current['open'] - current['low']) > eps:
            downside_res_vol = (current['volume'] / (current['open'] - current['low'])) * vol_ratio
        else:
            downside_res_vol = 0
            
        # Fractal Volume Asymmetry
        if downside_res_vol > eps:
            volume_asymmetry = (upside_abs_vol / downside_res_vol) * vol_ratio
        else:
            volume_asymmetry = 0
            
        # Fractal Trade Size
        if current['volume'] > eps:
            fractal_trade_size = (current['amount'] / current['volume']) * vol_ratio
        else:
            fractal_trade_size = 0
            
        # Fractal Trade Momentum
        if prev1['volume'] > eps:
            prev_trade_size = prev1['amount'] / prev1['volume']
            if prev_trade_size > eps:
                trade_momentum = ((current['amount'] / current['volume']) / prev_trade_size - 1) * vol_ratio
            else:
                trade_momentum = 0
        else:
            trade_momentum = 0
            
        # High Volatility Absorption
        high_vol_absorption = current['volume'] * (current['high'] - current['low']) * vol_ratio
        
        # Volume-Range Coherence
        if (prev1['volume'] * (prev1['high'] - prev1['low'])) > eps:
            volume_range_coherence = ((current['volume'] * (current['high'] - current['low'])) / 
                                    (prev1['volume'] * (prev1['high'] - prev1['low']))) * vol_ratio
        else:
            volume_range_coherence = 0
            
        # 4. Fractal Entropy Construction
        # Price Fractal Entropy
        dominance_measures = [
            abs(current['close'] - current['open']) / (current['high'] - current['low'] + eps),
            abs(current['high'] - current['low']) / (prev1['high'] - prev1['low'] + eps),
            vol_ratio
        ]
        
        price_entropy = 0
        for dom in dominance_measures:
            if dom > 0:
                price_entropy -= dom * np.log(dom + eps)
        price_entropy *= vol_ratio
        
        # Volume Fractal Entropy
        if vol_ratio > 0:
            volume_entropy = -(vol_ratio * np.log(abs(vol_ratio) + eps)) * vol_ratio
        else:
            volume_entropy = 0
            
        # 5. Fractal Breakout Efficiency
        # Upside Breakout
        upside_breakout = (current['high'] / (prev1['high'] + eps) - 1) * vol_ratio
        
        # Downside Breakout
        downside_breakout = (current['low'] / (prev1['low'] + eps) - 1) * vol_ratio
        
        # Price Breakout Asymmetry
        price_breakout_asymmetry = (upside_breakout - downside_breakout) * vol_ratio
        
        # Volume-Breakout Alignment
        volume_breakout_alignment = (upside_breakout - downside_breakout) * vol_ratio * vol_ratio
        
        # Clean Fractal Momentum
        clean_momentum = (current['close'] / (prev1['close'] + eps) - 1) * vol_ratio
        
        # Fractal Efficiency Ratio
        if (current['high'] - current['low']) > eps:
            efficiency_ratio = (clean_momentum / ((current['high'] - current['low']) / (prev1['close'] + eps))) * vol_ratio
        else:
            efficiency_ratio = 0
            
        # 6. Cross-Fractal Validation
        # Efficiency-Volume Divergence
        prev_vol_ratio = prev1['volume'] / (df.iloc[i-2]['volume'] + eps)
        efficiency_volume_div = (np.sign(fractal_efficiency - prev_efficiency) * 
                               np.sign(vol_ratio - 1)) * vol_ratio
        
        # Rejection-Velocity Alignment
        rejection_velocity = np.sign(net_asymmetry) * np.sign(vol_ratio - 1) * vol_ratio
        
        # Fractal Range Expansion/Contraction
        range_ratio = (current['high'] - current['low']) / (prev1['high'] - prev1['low'] + eps)
        range_expansion = (range_ratio > 1.2) * vol_ratio
        range_contraction = (range_ratio < 0.8) * vol_ratio
        
        # Fractal Mean Reversion
        if (current['high'] - current['low']) > eps:
            mean_reversion = (1 - abs(current['close'] - prev1['close']) / (current['high'] - current['low'])) * vol_ratio
        else:
            mean_reversion = 0
            
        # Momentum Consistency
        momentum_signs = []
        for j in range(max(0, i-2), i+1):
            if j >= 1:
                close_j = df.iloc[j]['close']
                close_prev = df.iloc[j-1]['close']
                vol_j = df.iloc[j]['volume']
                vol_prev_j = df.iloc[j-1]['volume']
                mom_j = (close_j / (close_prev + eps) - 1) * (vol_j / (vol_prev_j + eps))
                momentum_signs.append(np.sign(mom_j))
        
        if len(momentum_signs) >= 2:
            mom_consistency_count = sum(1 for k in range(1, len(momentum_signs)) if momentum_signs[k] == momentum_signs[k-1])
            momentum_consistency = (mom_consistency_count / (len(momentum_signs) - 1)) * vol_ratio
        else:
            momentum_consistency = 0
            
        # 7. Regime-Adaptive Alpha Construction
        # Core Velocity Components
        asymmetric_absorption_velocity = net_asymmetry * efficiency_momentum
        volume_fractal_momentum = clean_momentum * (vol_ratio - 1)
        trade_size_velocity = trade_momentum * efficiency_ratio
        
        # Fractal Confirmed Signals
        microstructure_velocity = asymmetric_absorption_velocity * efficiency_volume_div
        volume_aligned_absorption = volume_fractal_momentum * rejection_velocity
        range_enhanced_absorption = price_breakout_asymmetry * (range_expansion - range_contraction)
        
        # Final Alpha Construction
        primary_factor = microstructure_velocity * volume_range_coherence
        secondary_factor = volume_aligned_absorption * momentum_consistency
        
        # Composite Alpha with regime weighting
        regime_weight = 0.6  # Primary factor weight
        composite_alpha = (regime_weight * primary_factor + 
                          (1 - regime_weight) * secondary_factor + 
                          0.1 * range_enhanced_absorption)
        
        result.iloc[i] = composite_alpha
        
    # Fill initial values
    result = result.fillna(0)
    
    return result
