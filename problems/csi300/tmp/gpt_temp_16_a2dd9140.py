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
        if i < 10:  # Need at least 10 days of history
            result.iloc[i] = 0
            continue
            
        # Current day data
        high_t = df['high'].iloc[i]
        low_t = df['low'].iloc[i]
        open_t = df['open'].iloc[i]
        close_t = df['close'].iloc[i]
        volume_t = df['volume'].iloc[i]
        amount_t = df['amount'].iloc[i]
        
        # Historical data
        high_t3 = df['high'].iloc[i-3] if i >= 3 else high_t
        low_t3 = df['low'].iloc[i-3] if i >= 3 else low_t
        high_t8 = df['high'].iloc[i-8] if i >= 8 else high_t
        low_t8 = df['low'].iloc[i-8] if i >= 8 else low_t
        high_t1 = df['high'].iloc[i-1] if i >= 1 else high_t
        low_t1 = df['low'].iloc[i-1] if i >= 1 else low_t
        
        volume_t3 = df['volume'].iloc[i-3] if i >= 3 else volume_t
        amount_t3 = df['amount'].iloc[i-3] if i >= 3 else amount_t
        amount_t1 = df['amount'].iloc[i-1] if i >= 1 else amount_t
        amount_t8 = df['amount'].iloc[i-8] if i >= 8 else amount_t
        
        # Fractal Momentum Framework
        # Intraday Fractal Components
        morning_fractal_momentum = (high_t - open_t) / max(open_t - low_t, 1e-6)
        afternoon_fractal_momentum = (close_t - low_t) / max(high_t - close_t, 1e-6)
        intraday_fractal_shift = morning_fractal_momentum - afternoon_fractal_momentum
        
        # Historical intraday fractal shifts
        intraday_fractal_shift_t3 = 0
        intraday_fractal_shift_t8 = 0
        if i >= 3:
            high_t3 = df['high'].iloc[i-3]
            low_t3 = df['low'].iloc[i-3]
            open_t3 = df['open'].iloc[i-3]
            close_t3 = df['close'].iloc[i-3]
            morning_fractal_momentum_t3 = (high_t3 - open_t3) / max(open_t3 - low_t3, 1e-6)
            afternoon_fractal_momentum_t3 = (close_t3 - low_t3) / max(high_t3 - close_t3, 1e-6)
            intraday_fractal_shift_t3 = morning_fractal_momentum_t3 - afternoon_fractal_momentum_t3
        
        if i >= 8:
            high_t8 = df['high'].iloc[i-8]
            low_t8 = df['low'].iloc[i-8]
            open_t8 = df['open'].iloc[i-8]
            close_t8 = df['close'].iloc[i-8]
            morning_fractal_momentum_t8 = (high_t8 - open_t8) / max(open_t8 - low_t8, 1e-6)
            afternoon_fractal_momentum_t8 = (close_t8 - low_t8) / max(high_t8 - close_t8, 1e-6)
            intraday_fractal_shift_t8 = morning_fractal_momentum_t8 - afternoon_fractal_momentum_t8
        
        # Multi-Timeframe Fractal Momentum
        short_term_fractal_momentum = intraday_fractal_shift - intraday_fractal_shift_t3
        medium_term_fractal_momentum = intraday_fractal_shift - intraday_fractal_shift_t8
        fractal_momentum_convergence = np.sign(short_term_fractal_momentum) * np.sign(medium_term_fractal_momentum)
        
        # Asymmetric Volume-Fractal Dynamics
        morning_volume_fractal = volume_t * (high_t - open_t)
        afternoon_volume_fractal = volume_t * (close_t - low_t)
        volume_fractal_asymmetry = morning_volume_fractal / max(afternoon_volume_fractal, 1e-6)
        
        amount_fractal_persistence = amount_t / max(amount_t3, 1e-6)
        amount_fractal_momentum = (amount_t / max(amount_t1, 1e-6)) - 1
        net_volume_fractal_bias = (morning_volume_fractal * amount_fractal_persistence) - (afternoon_volume_fractal * amount_fractal_momentum)
        
        # Multi-Scale Convergence-Divergence Detection
        # Short-Term Microstructure (3-day)
        price_microstructure_conv = (high_t - low_t) - (high_t3 - low_t3)
        volume_microstructure_conv = volume_t - volume_t3
        short_term_microstructure = np.sign(price_microstructure_conv) * np.sign(volume_microstructure_conv) * abs(price_microstructure_conv - volume_microstructure_conv)
        
        # Medium-Term Microstructure (8-day)
        # Historical volume fractal asymmetry
        volume_fractal_asymmetry_t8 = 0
        if i >= 8:
            volume_t8 = df['volume'].iloc[i-8]
            high_t8 = df['high'].iloc[i-8]
            low_t8 = df['low'].iloc[i-8]
            open_t8 = df['open'].iloc[i-8]
            close_t8 = df['close'].iloc[i-8]
            morning_volume_fractal_t8 = volume_t8 * (high_t8 - open_t8)
            afternoon_volume_fractal_t8 = volume_t8 * (close_t8 - low_t8)
            volume_fractal_asymmetry_t8 = morning_volume_fractal_t8 / max(afternoon_volume_fractal_t8, 1e-6)
        
        fractal_microstructure_conv = intraday_fractal_shift - intraday_fractal_shift_t8
        volume_fractal_conv = volume_fractal_asymmetry - volume_fractal_asymmetry_t8
        medium_term_microstructure = np.sign(fractal_microstructure_conv) * np.sign(volume_fractal_conv) * abs(fractal_microstructure_conv - volume_fractal_conv)
        
        microstructure_convergence_consistency = np.sign(short_term_microstructure) * np.sign(medium_term_microstructure) * min(abs(short_term_microstructure), abs(medium_term_microstructure))
        
        # Fractal Regime Detection
        morning_fractal_expansion = morning_fractal_momentum > afternoon_fractal_momentum
        volume_fractal_expansion = morning_volume_fractal > afternoon_volume_fractal
        fractal_expansion_intensity = abs(morning_fractal_momentum - afternoon_fractal_momentum) * abs(morning_volume_fractal - afternoon_volume_fractal)
        
        high_fractal_regime = intraday_fractal_shift * fractal_expansion_intensity
        low_fractal_regime = intraday_fractal_shift / max(fractal_expansion_intensity, 1e-6)
        fractal_regime_switch = high_fractal_regime - low_fractal_regime
        
        # Adaptive Signal Integration
        # TrueRange calculation
        true_range = max(high_t - low_t, abs(high_t - close_t), abs(low_t - close_t))
        fractal_context = ((high_t - low_t) / max(true_range, 1e-6)) 
        
        # Count positive shifts in last 3 days
        positive_shift_count = 0
        for j in range(max(0, i-3), i):
            if j > 0:
                high_j = df['high'].iloc[j]
                low_j = df['low'].iloc[j]
                open_j = df['open'].iloc[j]
                close_j = df['close'].iloc[j]
                morning_fractal_j = (high_j - open_j) / max(open_j - low_j, 1e-6)
                afternoon_fractal_j = (close_j - low_j) / max(high_j - close_j, 1e-6)
                intraday_fractal_shift_j = morning_fractal_j - afternoon_fractal_j
                
                high_j1 = df['high'].iloc[j-1]
                low_j1 = df['low'].iloc[j-1]
                open_j1 = df['open'].iloc[j-1]
                close_j1 = df['close'].iloc[j-1]
                morning_fractal_j1 = (high_j1 - open_j1) / max(open_j1 - low_j1, 1e-6)
                afternoon_fractal_j1 = (close_j1 - low_j1) / max(high_j1 - close_j1, 1e-6)
                intraday_fractal_shift_j1 = morning_fractal_j1 - afternoon_fractal_j1
                
                if intraday_fractal_shift_j > intraday_fractal_shift_j1:
                    positive_shift_count += 1
        
        fractal_context += positive_shift_count
        
        # Momentum Acceleration
        momentum_acceleration = short_term_fractal_momentum
        if i >= 1:
            high_t1 = df['high'].iloc[i-1]
            low_t1 = df['low'].iloc[i-1]
            open_t1 = df['open'].iloc[i-1]
            close_t1 = df['close'].iloc[i-1]
            morning_fractal_t1 = (high_t1 - open_t1) / max(open_t1 - low_t1, 1e-6)
            afternoon_fractal_t1 = (close_t1 - low_t1) / max(high_t1 - close_t1, 1e-6)
            intraday_fractal_shift_t1 = morning_fractal_t1 - afternoon_fractal_t1
            short_term_fractal_momentum_t1 = intraday_fractal_shift_t1 - intraday_fractal_shift_t3
            momentum_acceleration += short_term_fractal_momentum_t1
        
        if i >= 2:
            high_t2 = df['high'].iloc[i-2]
            low_t2 = df['low'].iloc[i-2]
            open_t2 = df['open'].iloc[i-2]
            close_t2 = df['close'].iloc[i-2]
            morning_fractal_t2 = (high_t2 - open_t2) / max(open_t2 - low_t2, 1e-6)
            afternoon_fractal_t2 = (close_t2 - low_t2) / max(high_t2 - close_t2, 1e-6)
            intraday_fractal_shift_t2 = morning_fractal_t2 - afternoon_fractal_t2
            short_term_fractal_momentum_t2 = intraday_fractal_shift_t2 - intraday_fractal_shift_t3
            momentum_acceleration += short_term_fractal_momentum_t2
        
        # Regime Adaptive Weight
        if volume_fractal_asymmetry > 1.2 and microstructure_convergence_consistency > 0:
            regime_adaptive_weight = 1.4
        elif volume_fractal_asymmetry < 0.8 and abs(short_term_microstructure - medium_term_microstructure) > 0:
            regime_adaptive_weight = 0.6
        else:
            regime_adaptive_weight = (amount_t / max(amount_t3, 1e-6)) - (amount_t / max(amount_t8, 1e-6))
        
        # Fractal Convergence Confirmation
        # Calculate averages for intraday fractal shift
        avg_3day = 0
        count_3day = 0
        for j in range(max(0, i-3), i+1):
            high_j = df['high'].iloc[j]
            low_j = df['low'].iloc[j]
            open_j = df['open'].iloc[j]
            close_j = df['close'].iloc[j]
            morning_fractal_j = (high_j - open_j) / max(open_j - low_j, 1e-6)
            afternoon_fractal_j = (close_j - low_j) / max(high_j - close_j, 1e-6)
            intraday_fractal_shift_j = morning_fractal_j - afternoon_fractal_j
            avg_3day += intraday_fractal_shift_j
            count_3day += 1
        avg_3day = avg_3day / max(count_3day, 1)
        
        avg_10day = 0
        count_10day = 0
        for j in range(max(0, i-10), i+1):
            high_j = df['high'].iloc[j]
            low_j = df['low'].iloc[j]
            open_j = df['open'].iloc[j]
            close_j = df['close'].iloc[j]
            morning_fractal_j = (high_j - open_j) / max(open_j - low_j, 1e-6)
            afternoon_fractal_j = (close_j - low_j) / max(high_j - close_j, 1e-6)
            intraday_fractal_shift_j = morning_fractal_j - afternoon_fractal_j
            avg_10day += intraday_fractal_shift_j
            count_10day += 1
        avg_10day = avg_10day / max(count_10day, 1)
        
        fractal_convergence_confirmation = (intraday_fractal_shift / max(avg_3day, 1e-6)) * (intraday_fractal_shift / max(avg_10day, 1e-6))
        
        # Final Alpha Generation
        alpha_value = (net_volume_fractal_bias * fractal_regime_switch * 
                      (short_term_microstructure + medium_term_microstructure) * 
                      momentum_acceleration * fractal_momentum_convergence * 
                      regime_adaptive_weight) * fractal_convergence_confirmation * \
                      ((close_t - open_t) * volume_t / max(high_t1 - low_t1, 1e-6))
        
        result.iloc[i] = alpha_value
    
    return result
