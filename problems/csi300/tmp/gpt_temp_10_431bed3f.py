import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 8:  # Need at least 8 days for calculations
            result.iloc[i] = 0
            continue
            
        # Current day data
        open_t = df['open'].iloc[i]
        high_t = df['high'].iloc[i]
        low_t = df['low'].iloc[i]
        close_t = df['close'].iloc[i]
        volume_t = df['volume'].iloc[i]
        amount_t = df['amount'].iloc[i]
        
        # Multi-Scale Momentum Microstructure
        # Intraday Momentum Microstructure
        morning_momentum_micro = ((close_t - open_t) / (high_t - low_t + 0.001)) * (open_t - low_t)
        afternoon_momentum_micro = ((close_t - open_t) / (high_t - low_t + 0.001)) * (high_t - close_t)
        intraday_momentum_asymmetry = morning_momentum_micro - afternoon_momentum_micro
        
        # Short-Term Momentum Microstructure (3-day window)
        high_3d = max(df['high'].iloc[i-3:i+1])
        low_3d = min(df['low'].iloc[i-3:i+1])
        short_term_morning_momentum = ((close_t - open_t) / (high_3d - low_3d + 0.001)) * (open_t - low_3d)
        short_term_afternoon_momentum = ((close_t - open_t) / (high_3d - low_3d + 0.001)) * (high_3d - close_t)
        short_term_momentum_asymmetry = short_term_morning_momentum - short_term_afternoon_momentum
        
        # Medium-Term Momentum Microstructure (8-day window)
        high_8d = max(df['high'].iloc[i-8:i+1])
        low_8d = min(df['low'].iloc[i-8:i+1])
        medium_term_morning_momentum = ((close_t - open_t) / (high_8d - low_8d + 0.001)) * (open_t - low_8d)
        medium_term_afternoon_momentum = ((close_t - open_t) / (high_8d - low_8d + 0.001)) * (high_8d - close_t)
        medium_term_momentum_asymmetry = medium_term_morning_momentum - medium_term_afternoon_momentum
        
        # Momentum Microstructure Cascade
        momentum_micro_cascade = intraday_momentum_asymmetry * short_term_momentum_asymmetry * medium_term_momentum_asymmetry
        momentum_horizon_resonance = np.sign(short_term_momentum_asymmetry - medium_term_momentum_asymmetry) * np.sign(intraday_momentum_asymmetry - short_term_momentum_asymmetry)
        momentum_micro_resonance = momentum_micro_cascade * momentum_horizon_resonance
        
        # Volume-Momentum Microstructure Dynamics
        morning_volume_momentum = volume_t * morning_momentum_micro
        afternoon_volume_momentum = volume_t * afternoon_momentum_micro
        volume_momentum_differential = morning_volume_momentum - afternoon_volume_momentum
        
        # Volume Momentum Transmission
        volume_momentum_ratio = volume_t / (abs(close_t - open_t) + 0.001)
        volume_momentum_velocity = volume_momentum_ratio - (df['volume'].iloc[i-1] / (abs(df['close'].iloc[i-1] - df['open'].iloc[i-1]) + 0.001))
        volume_momentum_acceleration = volume_momentum_velocity - ((df['volume'].iloc[i-1] / (abs(df['close'].iloc[i-1] - df['open'].iloc[i-1]) + 0.001)) - (df['volume'].iloc[i-2] / (abs(df['close'].iloc[i-2] - df['open'].iloc[i-2]) + 0.001)))
        
        # Volume Momentum Alignment
        prev_open = df['open'].iloc[i-1]
        prev_high = df['high'].iloc[i-1]
        prev_low = df['low'].iloc[i-1]
        prev_close = df['close'].iloc[i-1]
        prev_volume = df['volume'].iloc[i-1]
        
        prev_morning_momentum = ((prev_close - prev_open) / (prev_high - prev_low + 0.001)) * (prev_open - prev_low)
        prev_afternoon_momentum = ((prev_close - prev_open) / (prev_high - prev_low + 0.001)) * (prev_high - prev_close)
        prev_volume_momentum_diff = prev_volume * prev_morning_momentum - prev_volume * prev_afternoon_momentum
        
        volume_momentum_gradient = volume_momentum_differential - prev_volume_momentum_diff
        price_momentum_gradient = intraday_momentum_asymmetry - (prev_morning_momentum - prev_afternoon_momentum)
        volume_price_momentum_alignment = np.sign(volume_momentum_gradient) * np.sign(price_momentum_gradient) * volume_momentum_differential
        
        # Amount-Driven Momentum Microstructure
        momentum_trade_size = amount_t / (volume_t + 0.001)
        trade_size_momentum_ratio = momentum_trade_size / (df['amount'].iloc[i-1] / (df['volume'].iloc[i-1] + 0.001) + 0.001)
        trade_size_momentum_volatility = abs(momentum_trade_size - (df['amount'].iloc[i-1] / (df['volume'].iloc[i-1] + 0.001))) / (df['amount'].iloc[i-1] / (df['volume'].iloc[i-1] + 0.001) + 0.001)
        
        amount_per_momentum_unit = amount_t / (abs(close_t - open_t) + 0.001)
        volume_per_momentum_unit = volume_t / (abs(close_t - open_t) + 0.001)
        amount_volume_momentum_efficiency = amount_per_momentum_unit * volume_per_momentum_unit * ((close_t - open_t) ** 2) / (((open_t - df['close'].iloc[i-1]) ** 2) + ((close_t - open_t) ** 2) + 0.001)
        
        large_trade_momentum_pressure = amount_t * ((close_t - open_t) / (high_t - low_t + 0.001)) * (open_t - low_t)
        small_trade_momentum_pressure = volume_t * ((close_t - open_t) / (high_t - low_t + 0.001)) * (high_t - close_t)
        trade_size_momentum_divergence = large_trade_momentum_pressure - small_trade_momentum_pressure
        
        # Momentum-Volume Pattern Recognition
        # Volume Cluster Microstructure
        consecutive_high_volume = 0
        for j in range(max(i-2, 0), i+1):
            if j >= 5:
                avg_volume_5d = sum(df['volume'].iloc[j-5:j]) / 5
                if df['volume'].iloc[j] > avg_volume_5d:
                    consecutive_high_volume += 1
        
        volume_breakout_micro = volume_t / (max(df['volume'].iloc[max(i-5, 0):i]) + 0.001)
        volume_cluster_momentum = consecutive_high_volume * volume_breakout_micro * (close_t - df['close'].iloc[i-1])
        
        # Momentum Persistence Patterns
        intraday_momentum_persistence = 0
        volume_trend_consistency = 0
        for j in range(max(i-4, 0), i+1):
            if df['close'].iloc[j] > df['close'].iloc[j-1]:
                intraday_momentum_persistence += 1
            if df['volume'].iloc[j] > df['volume'].iloc[j-1]:
                volume_trend_consistency += 1
        
        momentum_volume_alignment = intraday_momentum_persistence * volume_trend_consistency * np.sign(close_t - df['close'].iloc[i-1])
        
        # Price-Volume Divergence Micro
        negative_divergence_micro = ((close_t - df['close'].iloc[i-5]) / (df['close'].iloc[i-5] + 0.001)) - ((volume_t - df['volume'].iloc[i-5]) / (df['volume'].iloc[i-5] + 0.001))
        
        price_change_sum = 0
        for j in range(max(i-4, 0), i+1):
            price_change_sum += abs(df['close'].iloc[j] - df['close'].iloc[j-1])
        
        volume_price_efficiency_micro = ((close_t - df['close'].iloc[i-5]) / (price_change_sum + 0.001)) * (volume_t / (df['volume'].iloc[i-5] + 0.001))
        
        prev_neg_divergence = ((df['close'].iloc[i-1] - df['close'].iloc[i-6]) / (df['close'].iloc[i-6] + 0.001)) - ((df['volume'].iloc[i-1] - df['volume'].iloc[i-6]) / (df['volume'].iloc[i-6] + 0.001))
        divergence_reversal_micro = np.sign(df['close'].iloc[i-1] - df['close'].iloc[i-2]) * (negative_divergence_micro - prev_neg_divergence)
        
        # Adaptive Momentum Micro Synchronization
        volume_momentum_volatility_weighted = volume_momentum_differential / (high_t - low_t + 0.001)
        directional_momentum_volatility_weighted = trade_size_momentum_divergence * (high_t - low_t)
        
        recent_volatility = (high_t - low_t) + (df['high'].iloc[i-1] - df['low'].iloc[i-1]) + (df['high'].iloc[i-2] - df['low'].iloc[i-2])
        momentum_efficiency_volatility_adjusted = amount_volume_momentum_efficiency / (recent_volatility + 0.001)
        
        momentum_resonance_volume_weighted = momentum_micro_resonance * volume_t
        trade_size_momentum_volume_weighted = trade_size_momentum_divergence / (volume_t + 0.001)
        volume_timing_momentum_adjusted = volume_momentum_differential * ((volume_t / df['volume'].iloc[i-1]) - 1)
        
        # Momentum Regime Micro Synchronization
        high_momentum_micro_signal = ((close_t - open_t) / (high_8d - low_8d + 0.001)) * volume_momentum_differential * large_trade_momentum_pressure
        
        recent_price_changes = sum([abs(df['close'].iloc[j] - df['open'].iloc[j]) for j in range(max(i-5, 0), i)])
        low_momentum_micro_signal = ((close_t - open_t) / (recent_price_changes + 0.001)) * volume_momentum_differential * small_trade_momentum_pressure
        
        momentum_adaptive_micro_signal = high_momentum_micro_signal if ((close_t - open_t) / (high_3d - low_3d + 0.001)) > 0 else low_momentum_micro_signal
        
        # Composite Microstructure Alpha Construction
        efficiency_momentum_micro = amount_volume_momentum_efficiency * volume_momentum_differential
        directional_momentum_micro = trade_size_momentum_divergence * volume_momentum_differential
        pattern_momentum_micro = volume_cluster_momentum * momentum_volume_alignment
        
        base_micro_resonance = efficiency_momentum_micro * directional_momentum_micro * pattern_momentum_micro
        volatility_volume_micro_enhanced = base_micro_resonance * momentum_adaptive_micro_signal
        
        # Count positive momentum days in last 4 days
        positive_momentum_count = 0
        for j in range(max(i-3, 0), i+1):
            if (df['close'].iloc[j] - df['open'].iloc[j]) > (df['close'].iloc[j-1] - df['open'].iloc[j-1]):
                positive_momentum_count += 1
        
        momentum_volume_microstructure_resonance_alpha = volatility_volume_micro_enhanced * ((volume_t / df['volume'].iloc[i-1]) - 1) * (positive_momentum_count / 4)
        
        result.iloc[i] = momentum_volume_microstructure_resonance_alpha
    
    return result
