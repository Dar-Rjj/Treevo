import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic rolling statistics
    df['prev_close'] = df['close'].shift(1)
    df['true_range'] = np.maximum(df['high'] - df['low'], 
                                 np.maximum(abs(df['high'] - df['prev_close']), 
                                           abs(df['low'] - df['prev_close'])))
    
    # Rolling windows
    for i in range(len(df)):
        if i < 21:  # Need enough data for calculations
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]
        
        # Volatility Regime & Liquidity Integration
        # Gap-adjusted volatility efficiency
        high_t = current_data['high'].iloc[-1]
        low_t = current_data['low'].iloc[-1]
        close_t = current_data['close'].iloc[-1]
        open_t = current_data['open'].iloc[-1]
        volume_t = current_data['volume'].iloc[-1]
        prev_close = current_data['prev_close'].iloc[-1]
        
        gap_adj_vol_eff = ((high_t - low_t) / 
                          (max(high_t, prev_close) - min(low_t, prev_close)) * 
                          (volume_t / max(abs(close_t - prev_close), 0.001)))
        
        # Asymmetric range expansion
        asym_range_exp = (((high_t - close_t) / max(close_t - low_t, 0.001)) * 
                         abs(open_t / max(prev_close, 0.001) - 1))
        
        # Volatility spillover intensity
        tr_5day = current_data['true_range'].iloc[-5:].mean()
        tr_21day = current_data['true_range'].iloc[-21:].mean()
        avg_range_20day = (current_data['high'].iloc[-21:-1] - current_data['low'].iloc[-21:-1]).mean()
        vol_spillover = (tr_5day / max(tr_21day, 0.001)) * ((high_t - low_t) / max(avg_range_20day, 0.001))
        
        # Multi-period regime transitions
        # Volatility-liquidity persistence
        vol_liquidity_persistence = 0
        for j in range(min(10, i)):
            idx = i - j
            daily_range = current_data['high'].iloc[idx] - current_data['low'].iloc[idx]
            avg_range_5 = (current_data['high'].iloc[max(0, idx-4):idx+1] - 
                          current_data['low'].iloc[max(0, idx-4):idx+1]).mean()
            avg_vol_5 = current_data['volume'].iloc[max(0, idx-4):idx+1].mean()
            
            if (daily_range > avg_range_5) and (current_data['volume'].iloc[idx] > avg_vol_5):
                vol_liquidity_persistence += 1
        
        # Low-volatility compression
        low_vol_compression = 0
        for j in range(min(21, i)):
            idx = i - j
            daily_range = current_data['high'].iloc[idx] - current_data['low'].iloc[idx]
            avg_range_21 = (current_data['high'].iloc[max(0, idx-20):idx+1] - 
                           current_data['low'].iloc[max(0, idx-20):idx+1]).mean()
            avg_vol_21 = current_data['volume'].iloc[max(0, idx-20):idx+1].mean()
            
            if (daily_range < avg_range_21) and (current_data['volume'].iloc[idx] < avg_vol_21):
                low_vol_compression += 1
        
        # Price Anchoring & Momentum Dynamics
        # Gap-breakout proximity
        high_4day = current_data['high'].iloc[-5:].max()
        gap_breakout = ((high_t - high_4day) / max(high_t, 0.001)) * abs(open_t / max(prev_close, 0.001) - 1)
        
        # Support-level efficiency
        low_4day = current_data['low'].iloc[-5:].min()
        support_eff = (((low_4day - low_t) / max(low_t, 0.001)) * 
                      ((close_t - open_t) / max(high_t - low_t, 0.001)))
        
        # Anchoring momentum bias
        high_5day = current_data['high'].iloc[-6:-1].max()
        low_5day = current_data['low'].iloc[-6:-1].min()
        close_5day_ago = current_data['close'].iloc[-6]
        dist_high = (high_5day - close_t) / max(high_5day, 0.001)
        dist_low = (close_t - low_5day) / max(close_t, 0.001)
        anchor_momentum = (dist_high - dist_low) * (close_t / max(close_5day_ago, 0.001) - 1)
        
        # Volume-Flow Microstructure Synthesis
        # Gap-volume asymmetry
        if close_t > prev_close:
            up_volume = volume_t
            down_volume = 0
        elif close_t < prev_close:
            up_volume = 0
            down_volume = volume_t
        else:
            up_volume = volume_t / 2
            down_volume = volume_t / 2
        
        gap_vol_asym = ((up_volume - down_volume) / max(up_volume + down_volume, 0.001))
        
        # Volume clustering efficiency
        avg_vol_5 = current_data['volume'].iloc[-5:].mean()
        vol_clustering = volume_t / max(avg_vol_5, 0.001)
        
        # Flow persistence quality
        flow_persistence = 0
        consecutive_count = 0
        for j in range(min(10, i)):
            idx = i - j
            avg_vol_5_local = current_data['volume'].iloc[max(0, idx-4):idx+1].mean()
            if current_data['volume'].iloc[idx] > avg_vol_5_local:
                consecutive_count += 1
            else:
                break
        flow_persistence = consecutive_count
        
        # Regime-Adaptive Signal Construction
        # Volatility regime detection
        vol_regime_change = tr_5day / max(tr_21day, 0.001)
        avg_vol_5_current = current_data['volume'].iloc[-5:].mean()
        
        if vol_regime_change > 1 and volume_t > avg_vol_5_current:
            # High-volatility regime
            regime_signal = gap_breakout * vol_spillover * gap_vol_asym
        else:
            # Low-volatility regime  
            regime_signal = anchor_momentum * low_vol_compression/21.0 * ((high_t - low_t) / max(avg_range_20day, 0.001))
        
        # Transition smoothing
        transition_smoothing = regime_signal * (1 - abs(vol_regime_change - 1)) * vol_clustering
        
        # Microstructure Quality Enhancement
        # Flow-momentum multiplier
        microstructure_quality = gap_vol_asym * vol_clustering * flow_persistence/10.0
        institutional_flow = vol_clustering * flow_persistence/10.0
        
        flow_momentum_mult = transition_smoothing * microstructure_quality * institutional_flow
        
        # Final composite factor
        composite_factor = (flow_momentum_mult * 
                          gap_adj_vol_eff * 
                          asym_range_exp * 
                          support_eff * 
                          anchor_momentum)
        
        result.iloc[i] = composite_factor
    
    # Clean infinite values and normalize
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.fillna(0)
    
    return result
