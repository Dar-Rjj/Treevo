import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    factor_values = pd.Series(index=df.index, dtype=float)
    
    for i in range(10, len(df)):
        current_data = df.iloc[:i+1]
        
        # Multi-Timeframe Momentum Spectrum
        # Momentum Acceleration Profile
        if i >= 10:
            ret_2d = (current_data['close'].iloc[i] / current_data['close'].iloc[i-2]) - 1
            ret_5d = (current_data['close'].iloc[i] / current_data['close'].iloc[i-5]) - 1
            ret_10d = (current_data['close'].iloc[i] / current_data['close'].iloc[i-10]) - 1
            momentum_accel = (ret_2d - ret_5d) + (ret_5d - ret_10d)
        else:
            momentum_accel = 0
        
        # Volatility-Adjusted Momentum Blend
        if i >= 14:
            ret_3d = (current_data['close'].iloc[i] / current_data['close'].iloc[i-3]) - 1
            ret_7d = (current_data['close'].iloc[i] / current_data['close'].iloc[i-7]) - 1
            returns_15d = [(current_data['close'].iloc[j] / current_data['close'].iloc[j-1]) - 1 
                          for j in range(i-13, i+1)]
            vol_15d = np.std(returns_15d) if len(returns_15d) > 1 else 0.001
            vol_adj_momentum = (ret_3d + ret_7d) / (2 * vol_15d) if vol_15d != 0 else 0
        else:
            vol_adj_momentum = 0
        
        # Momentum Persistence Score
        if i >= 10:
            mom_5d = current_data['close'].iloc[i] / current_data['close'].iloc[i-5]
            mom_10d = current_data['close'].iloc[i] / current_data['close'].iloc[i-10]
            range_vols = []
            for j in range(max(4, i-4), i+1):
                if j >= 1:
                    daily_range = (current_data['high'].iloc[j] - current_data['low'].iloc[j]) / current_data['close'].iloc[j]
                    range_vols.append(daily_range)
            avg_range_vol = np.mean(range_vols) if range_vols else 0.001
            momentum_persistence = (mom_5d / mom_10d) / avg_range_vol if avg_range_vol != 0 else 0
        else:
            momentum_persistence = 0
        
        # Regime-Adaptive Volume Integration
        # Volume-Confirmed Momentum
        if i >= 5:
            price_momentum_5d = (current_data['close'].iloc[i] / current_data['close'].iloc[i-5]) - 1
            avg_volume_5d = np.mean([current_data['volume'].iloc[j] for j in range(i-4, i+1)])
            high_volume_regime = 1 if current_data['volume'].iloc[i] > avg_volume_5d else 0
            volume_confirmed = price_momentum_5d * high_volume_regime
        else:
            volume_confirmed = 0
        
        # Multi-Timeframe Volume Divergence
        if i >= 7:
            price_trend_3d = current_data['close'].iloc[i] / current_data['close'].iloc[i-3]
            price_trend_7d = current_data['close'].iloc[i] / current_data['close'].iloc[i-7]
            volume_trend_3d = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-3]
            volume_divergence = (price_trend_3d / price_trend_7d) - volume_trend_3d
        else:
            volume_divergence = 0
        
        # Volume-Regime Return Accumulation
        if i >= 3:
            volume_increase_regime = 1 if current_data['volume'].iloc[i] > current_data['volume'].iloc[i-1] else 0
            returns_3d = []
            regime_returns = []
            for j in range(max(2, i-2), i+1):
                if j >= 1:
                    daily_ret = (current_data['close'].iloc[j] / current_data['close'].iloc[j-1]) - 1
                    returns_3d.append(daily_ret)
                    regime_ind = 1 if current_data['volume'].iloc[j] > current_data['volume'].iloc[j-1] else 0
                    regime_returns.append(daily_ret * regime_ind)
            
            vol_3d = np.std(returns_3d) if len(returns_3d) > 1 else 0.001
            regime_accumulation = np.sum(regime_returns) / vol_3d if vol_3d != 0 else 0
        else:
            regime_accumulation = 0
        
        # Intraday Pattern Efficiency
        # Gap Filling Momentum
        if i >= 1:
            overnight_gap = (current_data['open'].iloc[i] - current_data['close'].iloc[i-1]) / current_data['close'].iloc[i-1]
            intraday_movement = (current_data['close'].iloc[i] - current_data['open'].iloc[i]) / current_data['open'].iloc[i]
            gap_filling = overnight_gap * intraday_movement * np.sign(overnight_gap)
        else:
            gap_filling = 0
        
        # Range Utilization Efficiency
        if i >= 5:
            price_movement = abs(current_data['close'].iloc[i] - current_data['open'].iloc[i])
            daily_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
            utilizations = []
            for j in range(max(4, i-4), i+1):
                if j >= 0:
                    pm = abs(current_data['close'].iloc[j] - current_data['open'].iloc[j])
                    dr = current_data['high'].iloc[j] - current_data['low'].iloc[j]
                    if dr != 0:
                        utilizations.append(pm / dr)
            avg_utilization = np.mean(utilizations) if utilizations else 0.001
            current_utilization = price_movement / daily_range if daily_range != 0 else 0
            range_efficiency = current_utilization / avg_utilization if avg_utilization != 0 else 0
        else:
            range_efficiency = 0
        
        # Session Momentum Divergence
        if i >= 5:
            morning_strength = (current_data['high'].iloc[i] - current_data['open'].iloc[i]) / current_data['open'].iloc[i]
            afternoon_strength = (current_data['close'].iloc[i] - current_data['low'].iloc[i]) / current_data['low'].iloc[i]
            returns_5d = [(current_data['close'].iloc[j] / current_data['close'].iloc[j-1]) - 1 
                         for j in range(max(4, i-4), i+1) if j >= 1]
            vol_5d = np.std(returns_5d) if len(returns_5d) > 1 else 0.001
            session_divergence = (morning_strength - afternoon_strength) / vol_5d if vol_5d != 0 else 0
        else:
            session_divergence = 0
        
        # Multi-Signal Convergence
        # Price-Volume Trend Alignment
        if i >= 7:
            price_trend_3d = current_data['close'].iloc[i] / current_data['close'].iloc[i-3]
            price_trend_7d = current_data['close'].iloc[i] / current_data['close'].iloc[i-7]
            volume_trend_3d = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-3]
            trend_alignment = (price_trend_3d / price_trend_7d) * volume_trend_3d
        else:
            trend_alignment = 0
        
        # Momentum Consistency Profile
        if i >= 10:
            mom_2d = (current_data['close'].iloc[i] / current_data['close'].iloc[i-2]) - 1
            mom_5d = (current_data['close'].iloc[i] / current_data['close'].iloc[i-5]) - 1
            mom_10d = (current_data['close'].iloc[i] / current_data['close'].iloc[i-10]) - 1
            momentum_consistency = (mom_2d * mom_5d) + (mom_5d * mom_10d)
        else:
            momentum_consistency = 0
        
        # Volatility-Weighted Signal Strength
        if i >= 10:
            ret_3d = (current_data['close'].iloc[i] / current_data['close'].iloc[i-3]) - 1
            ret_7d = (current_data['close'].iloc[i] / current_data['close'].iloc[i-7]) - 1
            volume_confirmation = np.sign(current_data['volume'].iloc[i] - current_data['volume'].iloc[i-3])
            returns_10d = [(current_data['close'].iloc[j] / current_data['close'].iloc[j-1]) - 1 
                          for j in range(i-9, i+1) if j >= 1]
            vol_10d = np.std(returns_10d) if len(returns_10d) > 1 else 0.001
            vol_weighted_signal = ((ret_3d + ret_7d) * volume_confirmation) / vol_10d if vol_10d != 0 else 0
        else:
            vol_weighted_signal = 0
        
        # Dynamic Support/Resistance Analysis
        # Volatility-Adjusted Breakout
        if i >= 10:
            resistance_10d = max([current_data['high'].iloc[j] for j in range(i-9, i+1)])
            break_strength = current_data['close'].iloc[i] / resistance_10d
            returns_5d = [(current_data['close'].iloc[j] / current_data['close'].iloc[j-1]) - 1 
                         for j in range(max(4, i-4), i+1) if j >= 1]
            vol_5d = np.std(returns_5d) if len(returns_5d) > 1 else 0.001
            volatility_breakout = (break_strength * current_data['volume'].iloc[i]) / vol_5d if vol_5d != 0 else 0
        else:
            volatility_breakout = 0
        
        # Support Bounce Efficiency
        if i >= 10:
            support_10d = min([current_data['low'].iloc[j] for j in range(i-9, i+1)])
            bounce_efficiency = current_data['close'].iloc[i] / support_10d
            returns_5d = [(current_data['close'].iloc[j] / current_data['close'].iloc[j-1]) - 1 
                         for j in range(max(4, i-4), i+1) if j >= 1]
            vol_5d = np.std(returns_5d) if len(returns_5d) > 1 else 0.001
            support_bounce = (bounce_efficiency * current_data['amount'].iloc[i]) / vol_5d if vol_5d != 0 else 0
        else:
            support_bounce = 0
        
        # Range Breakout Intensity
        if i >= 5:
            current_range = (current_data['high'].iloc[i] - current_data['low'].iloc[i]) / current_data['close'].iloc[i]
            ranges = []
            for j in range(max(4, i-4), i+1):
                if j >= 0:
                    daily_range = (current_data['high'].iloc[j] - current_data['low'].iloc[j]) / current_data['close'].iloc[j]
                    ranges.append(daily_range)
            avg_range = np.mean(ranges) if ranges else 0.001
            returns_5d = [(current_data['close'].iloc[j] / current_data['close'].iloc[j-1]) - 1 
                         for j in range(max(4, i-4), i+1) if j >= 1]
            vol_5d = np.std(returns_5d) if len(returns_5d) > 1 else 0.001
            range_breakout = (current_range / avg_range) / vol_5d if vol_5d != 0 else 0
        else:
            range_breakout = 0
        
        # Combine all factors with equal weighting
        all_factors = [
            momentum_accel, vol_adj_momentum, momentum_persistence,
            volume_confirmed, volume_divergence, regime_accumulation,
            gap_filling, range_efficiency, session_divergence,
            trend_alignment, momentum_consistency, vol_weighted_signal,
            volatility_breakout, support_bounce, range_breakout
        ]
        
        # Remove zeros (factors that couldn't be calculated due to insufficient data)
        valid_factors = [f for f in all_factors if f != 0]
        
        if valid_factors:
            factor_values.iloc[i] = np.mean(valid_factors)
        else:
            factor_values.iloc[i] = 0
    
    return factor_values
