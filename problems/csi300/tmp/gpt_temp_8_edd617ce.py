import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize output series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # Calculate required lookback periods
    for i in range(len(data)):
        if i < 10:  # Need at least 10 days of history
            alpha.iloc[i] = 0
            continue
            
        current_data = data.iloc[:i+1]
        
        # 1. Market Regime Identification via Entropy Analysis
        # Price Entropy Calculation
        close_prices = current_data['close'].values
        
        # Short-term price entropy (5-day)
        st_diffs = np.abs(close_prices[-1] - close_prices[-6:-1])
        if np.sum(st_diffs) > 0:
            st_probs = st_diffs / np.sum(st_diffs)
            st_entropy = -np.sum(st_probs * np.log(st_probs + 1e-10))
        else:
            st_entropy = 0
            
        # Medium-term price entropy (10-day)
        mt_diffs = np.abs(close_prices[-1] - close_prices[-11:-1])
        if np.sum(mt_diffs) > 0:
            mt_probs = mt_diffs / np.sum(mt_diffs)
            mt_entropy = -np.sum(mt_probs * np.log(mt_probs + 1e-10))
        else:
            mt_entropy = 0
            
        # Volume Distribution Entropy
        volumes = current_data['volume'].values
        highs = current_data['high'].values
        lows = current_data['low'].values
        
        # Volume concentration entropy (5-day)
        recent_volumes = volumes[-5:]
        if np.sum(recent_volumes) > 0:
            vol_probs = recent_volumes / np.sum(recent_volumes)
            vol_entropy = -np.sum(vol_probs * np.log(vol_probs + 1e-10))
        else:
            vol_entropy = 0
            
        # Volume-range entropy
        recent_ranges = highs[-5:] - lows[-5:]
        range_volume_ratios = recent_ranges / (volumes[-5:] + 1e-10)
        if np.sum(range_volume_ratios) > 0:
            range_probs = range_volume_ratios / np.sum(range_volume_ratios)
            range_entropy = -np.sum(range_probs * np.log(range_probs + 1e-10))
        else:
            range_entropy = 0
            
        # Combined Regime Classification
        price_entropy_avg = (st_entropy + mt_entropy) / 2
        volume_entropy_avg = (vol_entropy + range_entropy) / 2
        
        # Regime classification
        high_price_entropy = price_entropy_avg > np.percentile([st_entropy, mt_entropy], 60)
        high_volume_entropy = volume_entropy_avg > np.percentile([vol_entropy, range_entropy], 60)
        
        if high_price_entropy and high_volume_entropy:
            regime = 'chaotic'
        elif not high_price_entropy and not high_volume_entropy:
            regime = 'trending'
        else:
            regime = 'transitional'
            
        # 2. Multi-Timeframe Range Acceleration Analysis
        current_range = highs[-1] - lows[-1]
        
        # Range Momentum Components
        range_2 = highs[-3] - lows[-3] if i >= 2 else current_range
        range_5 = highs[-6] - lows[-6] if i >= 5 else current_range
        range_10 = highs[-11] - lows[-11] if i >= 10 else current_range
        
        st_momentum = (current_range / (range_2 + 1e-10)) - 1
        mt_momentum = (current_range / (range_5 + 1e-10)) - 1
        lt_momentum = (current_range / (range_10 + 1e-10)) - 1
        
        # Range Acceleration Calculation
        st_mt_accel = st_momentum - mt_momentum
        mt_lt_accel = mt_momentum - lt_momentum
        st_lt_accel = st_momentum - lt_momentum
        
        # Range Acceleration Regime
        avg_accel = (st_mt_accel + mt_lt_accel + st_lt_accel) / 3
        positive_accel = avg_accel > 0
        negative_accel = avg_accel < -0.1
        
        # 3. Microstructure Efficiency with Entropy Adaptation
        # Price Efficiency Measurement
        prev_close = close_prices[-2]
        current_open = current_data['open'].iloc[-1]
        
        movement_eff = abs(close_prices[-1] - prev_close) / max(
            current_range, 
            abs(highs[-1] - prev_close), 
            abs(lows[-1] - prev_close)
        )
        
        gap_eff = abs(current_open - prev_close) / (abs(close_prices[-1] - current_open) + 1e-10)
        intraday_eff = abs(close_prices[-1] - current_open) / (current_range + 1e-10)
        
        # Volume Efficiency Analysis
        volume_5ma = np.mean(volumes[-6:-1]) if len(volumes) >= 6 else volumes[-1]
        volume_concentration = volumes[-1] / (volume_5ma + 1e-10)
        volume_momentum = (volumes[-1] / (volumes[-6] + 1e-10)) - 1 if i >= 5 else 0
        volume_range_alignment = volumes[-1] / (current_range + 1e-10)
        
        # Efficiency score
        efficiency_score = (movement_eff + (1 - gap_eff) + intraday_eff) / 3
        
        # 4. Regime-Adaptive Signal Generation
        signal = 0
        
        if regime == 'chaotic':
            # High entropy regime: focus on mean-reversion
            if negative_accel and efficiency_score > 0.6:
                signal = -1.0  # Strong mean-reversion
            elif positive_accel and efficiency_score < 0.4:
                signal = 0.3   # Weak continuation
            elif volume_concentration > 2.0:
                signal = -0.5  # Regime change precursor
                
        elif regime == 'trending':
            # Low entropy regime: focus on trend-following
            if positive_accel and efficiency_score > 0.7:
                signal = 1.0   # Strong trend continuation
            elif negative_accel and efficiency_score > 0.6:
                signal = -0.8  # Trend exhaustion
            elif volume_range_alignment > np.percentile([volume_range_alignment], 70):
                signal *= 1.2  # Signal strength enhancement
                
        else:  # transitional
            # Mixed entropy with range acceleration bias
            if positive_accel:
                signal = 0.5   # Directional bias positive
            elif negative_accel:
                signal = -0.5  # Directional bias negative
                
        # 5. Volatility-Adaptive Component Integration
        # True Range calculation
        tr1 = current_range
        tr2 = abs(highs[-1] - prev_close)
        tr3 = abs(lows[-1] - prev_close)
        true_range = max(tr1, tr2, tr3)
        
        # Volatility trend
        if i >= 10:
            tr_5ma = np.mean([max(h-l, abs(h-pc), abs(l-pc)) 
                            for h, l, pc in zip(highs[-6:-1], lows[-6:-1], close_prices[-7:-2])])
            tr_10ma = np.mean([max(h-l, abs(h-pc), abs(l-pc)) 
                             for h, l, pc in zip(highs[-11:-1], lows[-11:-1], close_prices[-12:-2])])
            vol_trend = tr_5ma / (tr_10ma + 1e-10) - 1
        else:
            vol_trend = 0
            
        # Volatility-adaptive weighting
        high_volatility = true_range > np.percentile([tr1, tr2, tr3], 70)
        
        if high_volatility and regime == 'chaotic':
            signal *= 1.2  # Emphasize mean-reversion
        elif not high_volatility and regime == 'trending':
            signal *= 1.3  # Emphasize momentum
        elif high_volatility and regime == 'trending':
            signal *= 0.8  # Scale range acceleration
            
        # 6. Composite Alpha Factor Output
        alpha.iloc[i] = signal
        
    return alpha
