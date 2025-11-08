import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate returns for volatility calculations
    returns = data['close'].pct_change()
    
    for i in range(max(60, len(data))):
        if i < 60:  # Need sufficient data for calculations
            result.iloc[i] = 0
            continue
            
        current_date = data.index[i]
        
        # 1. Momentum Decay with Volume Persistence
        try:
            # Intraday Momentum
            price_momentum = (data['close'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
            intraday_range = (data['high'].iloc[i] - data['low'].iloc[i]) / data['close'].iloc[i-1]
            
            # Momentum Decay (assuming 5 days since signal)
            decay_factor = np.exp(-0.1386 * 5)
            volume_ratio = data['volume'].iloc[i] / data['volume'].iloc[i-1] if data['volume'].iloc[i-1] != 0 else 1
            
            # Recent Volatility (10-day)
            recent_returns = returns.iloc[i-9:i+1]
            volatility = recent_returns.std() if len(recent_returns) > 0 else 0.01
            
            momentum_factor = (price_momentum * decay_factor * volume_ratio) / max(volatility, 0.01)
        except:
            momentum_factor = 0
        
        # 2. Multi-Timeframe Convergence Factor
        try:
            # Short-term (3-day)
            st_price = data['close'].iloc[i] / data['close'].iloc[i-3] - 1 if data['close'].iloc[i-3] != 0 else 0
            st_volume = data['volume'].iloc[i] / data['volume'].iloc[i-3] - 1 if data['volume'].iloc[i-3] != 0 else 0
            
            # Medium-term (10-day)
            mt_price = data['close'].iloc[i] / data['close'].iloc[i-10] - 1 if data['close'].iloc[i-10] != 0 else 0
            mt_volume = data['volume'].iloc[i] / data['volume'].iloc[i-10] - 1 if data['volume'].iloc[i-10] != 0 else 0
            
            # Long-term (30-day)
            lt_price = data['close'].iloc[i] / data['close'].iloc[i-30] - 1 if data['close'].iloc[i-30] != 0 else 0
            lt_volume = data['volume'].iloc[i] / data['volume'].iloc[i-30] - 1 if data['volume'].iloc[i-30] != 0 else 0
            
            # Weighted convergence
            price_convergence = 0.5 * st_price + 0.3 * mt_price + 0.2 * lt_price
            volume_convergence = 0.5 * st_volume + 0.3 * mt_volume + 0.2 * lt_volume
            
            convergence_factor = price_convergence * volume_convergence
        except:
            convergence_factor = 0
        
        # 3. Regime-Adaptive Price-Volume Efficiency
        try:
            # Volatility Regime Detection
            short_vol = returns.iloc[i-9:i+1].std() if len(returns.iloc[i-9:i+1]) > 0 else 0.01
            long_vol = returns.iloc[i-59:i+1].std() if len(returns.iloc[i-59:i+1]) > 0 else 0.01
            vol_ratio = short_vol / max(long_vol, 0.01)
            
            # Price-Volume Efficiency
            high_low_range = data['high'].iloc[i] - data['low'].iloc[i]
            movement_efficiency = (data['close'].iloc[i] - data['open'].iloc[i]) / max(high_low_range, 0.0001)
            
            volume_mean = data['volume'].iloc[i-5:i].mean() if i >= 5 else data['volume'].iloc[i]
            volume_intensity = data['volume'].iloc[i] / max(volume_mean, 1)
            
            # Regime-Adaptive Combination
            if vol_ratio < 0.8:
                # Low volatility regime
                regime_factor = movement_efficiency * volume_intensity
            elif vol_ratio > 1.2:
                # High volatility regime
                regime_factor = movement_efficiency / max(volume_intensity, 0.1)
            else:
                # Normal volatility regime
                regime_factor = movement_efficiency * np.sign(volume_intensity - 1)
        except:
            regime_factor = 0
        
        # 4. Volume-Volatility Breakout Confidence
        try:
            # True Range
            tr1 = data['high'].iloc[i] - data['low'].iloc[i]
            tr2 = abs(data['high'].iloc[i] - data['close'].iloc[i-1])
            tr3 = abs(data['close'].iloc[i-1] - data['low'].iloc[i])
            true_range = max(tr1, tr2, tr3)
            
            # Breakout Threshold
            prev_tr = [max(data['high'].iloc[j] - data['low'].iloc[j], 
                          abs(data['high'].iloc[j] - data['close'].iloc[j-1]),
                      abs(data['close'].iloc[j-1] - data['low'].iloc[j])) for j in range(i-9, i)]
            threshold = np.mean(prev_tr) * 1.5 if len(prev_tr) > 0 else true_range
            
            # Volume Confirmation
            volume_surge = data['volume'].iloc[i] / max(data['volume'].iloc[i-19:i].mean(), 1)
            volatility_expansion = true_range / max(np.mean(prev_tr), 0.0001) if len(prev_tr) > 0 else 1
            
            # Breakout Strength
            breakout_strength = (true_range - threshold) / max(threshold, 0.0001)
            
            breakout_factor = breakout_strength * volume_surge * volatility_expansion
        except:
            breakout_factor = 0
        
        # 5. Order Flow Persistence Factor
        try:
            # Directional Flow
            flow_direction = np.sign(data['close'].iloc[i] - data['open'].iloc[i])
            flow_magnitude = data['amount'].iloc[i] * flow_direction
            
            # Persistence Tracking (simplified - look back 5 days)
            streak_length = 1
            cumulative_flow = flow_magnitude
            
            for j in range(1, min(6, i+1)):
                prev_direction = np.sign(data['close'].iloc[i-j] - data['open'].iloc[i-j])
                if prev_direction == flow_direction:
                    streak_length += 1
                    cumulative_flow += data['amount'].iloc[i-j] * prev_direction
                else:
                    break
            
            # Flow Signal
            persistence_score = cumulative_flow * streak_length
            total_amount = sum(data['amount'].iloc[i-streak_length+1:i+1])
            
            order_flow_factor = persistence_score / max(total_amount, 1)
        except:
            order_flow_factor = 0
        
        # 6. Adaptive Mean Reversion Factor
        try:
            # Price Deviation
            ma_5 = data['close'].iloc[i-4:i+1].mean()
            price_deviation = (data['close'].iloc[i] - ma_5) / max(data['close'].iloc[i], 0.0001)
            
            # Volume-Regime Adjustment
            volume_ratio_mr = data['volume'].iloc[i] / max(data['volume'].iloc[i-9:i].mean(), 1)
            volatility_context = returns.iloc[i-4:i+1].std() if len(returns.iloc[i-4:i+1]) > 0 else 0.01
            
            # Adaptive Signal
            if volume_ratio_mr > 1.2:
                adaptive_signal = -price_deviation * volume_ratio_mr
            elif volume_ratio_mr < 0.8:
                adaptive_signal = price_deviation / max(volume_ratio_mr, 0.1)
            else:
                adaptive_signal = price_deviation
            
            mean_reversion_factor = adaptive_signal / max(volatility_context, 0.01)
        except:
            mean_reversion_factor = 0
        
        # Combine all factors with equal weighting
        combined_factor = (momentum_factor + convergence_factor + regime_factor + 
                          breakout_factor + order_flow_factor + mean_reversion_factor) / 6
        
        result.iloc[i] = combined_factor
    
    return result
