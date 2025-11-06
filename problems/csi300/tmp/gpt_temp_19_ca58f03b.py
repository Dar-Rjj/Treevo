import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Timeframe Gap Detection
    # Short-term Fractal Gap
    short_term_gap = np.zeros(len(data))
    for i in range(1, len(data)):
        if data['high'].iloc[i] > data['high'].iloc[i-1] and data['low'].iloc[i] > data['low'].iloc[i-1]:
            short_term_gap[i] = data['high'].iloc[i] - data['low'].iloc[i-1]
        elif data['high'].iloc[i] < data['high'].iloc[i-1] and data['low'].iloc[i] < data['low'].iloc[i-1]:
            short_term_gap[i] = data['high'].iloc[i-1] - data['low'].iloc[i]
    
    # Medium-term Fractal Gap
    medium_term_gap = np.zeros(len(data))
    for i in range(5, len(data)):
        high_max = data['high'].iloc[i-5:i].max()
        low_min = data['low'].iloc[i-5:i].min()
        
        if data['open'].iloc[i] > high_max:
            medium_term_gap[i] = data['open'].iloc[i] - high_max
        elif data['open'].iloc[i] < low_min:
            medium_term_gap[i] = low_min - data['open'].iloc[i]
    
    # Gap Momentum Construction
    net_fractal_gap = np.zeros(len(data))
    gap_momentum = np.zeros(len(data))
    for i in range(5, len(data)):
        net_fractal_gap[i] = short_term_gap[i] - short_term_gap[i-5]
        if short_term_gap[i] != 0 and short_term_gap[i-5] != 0:
            sign_product = np.sign(short_term_gap[i]) * np.sign(medium_term_gap[i])
            price_change = (data['close'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
            gap_momentum[i] = net_fractal_gap[i] * sign_product * price_change
    
    # Efficiency-Weighted Volume Analysis
    volume_fractal_ratio = np.zeros(len(data))
    for i in range(1, len(data)):
        numerator = 0
        denominator = 0
        
        if data['high'].iloc[i] > data['high'].iloc[i-1] and data['volume'].iloc[i] > data['volume'].iloc[i-1]:
            numerator += data['volume'].iloc[i]
            denominator += data['volume'].iloc[i]
        
        if data['low'].iloc[i] < data['low'].iloc[i-1] and data['volume'].iloc[i] > data['volume'].iloc[i-1]:
            numerator -= data['volume'].iloc[i]
            denominator += data['volume'].iloc[i]
        
        if denominator != 0:
            volume_fractal_ratio[i] = numerator / denominator
    
    # Efficiency Momentum
    efficiency_momentum = np.zeros(len(data))
    for i in range(10, len(data)):
        # Short-term efficiency (5-day)
        price_range_5 = sum(data['high'].iloc[i-4:i+1] - data['low'].iloc[i-4:i+1])
        if price_range_5 != 0:
            eff_5 = abs(data['close'].iloc[i] - data['close'].iloc[i-5]) / price_range_5
        else:
            eff_5 = 0
        
        # Medium-term efficiency (10-day)
        price_range_10 = sum(data['high'].iloc[i-9:i+1] - data['low'].iloc[i-9:i+1])
        if price_range_10 != 0:
            eff_10 = abs(data['close'].iloc[i] - data['close'].iloc[i-10]) / price_range_10
        else:
            eff_10 = 0
        
        efficiency_momentum[i] = eff_5 - eff_10
    
    # Volume-Price Momentum Alignment
    volume_price_divergence = np.zeros(len(data))
    volume_price_convergence = np.zeros(len(data))
    
    for i in range(5, len(data)):
        # Volume-Price Divergence
        price_change_2 = data['close'].iloc[i] / data['close'].iloc[i-2] - 1
        price_change_5 = data['close'].iloc[i] / data['close'].iloc[i-5] - 1
        volume_change_2 = data['volume'].iloc[i] / data['volume'].iloc[i-2] - 1
        volume_change_5 = data['volume'].iloc[i] / data['volume'].iloc[i-5] - 1
        
        price_diff = price_change_2 - price_change_5
        volume_diff = volume_change_2 - volume_change_5
        
        denominator = abs(price_diff) + abs(volume_diff)
        if denominator != 0:
            volume_price_divergence[i] = (price_diff - volume_diff) / denominator
        
        # Volume-Price Convergence
        if data['high'].iloc[i] != data['low'].iloc[i]:
            amount_vol = data['amount'].iloc[i] / (data['high'].iloc[i] - data['low'].iloc[i])
        else:
            amount_vol = 0
            
        volume_price_convergence[i] = (np.sign(volume_price_divergence[i]) * 
                                     volume_fractal_ratio[i] * 
                                     efficiency_momentum[i] * 
                                     amount_vol)
    
    # Volatility-Regime Processing
    total_daily_volatility = np.zeros(len(data))
    volatility_multiplier = np.ones(len(data))
    
    for i in range(1, len(data)):
        # Total Daily Volatility
        mid_price = (data['open'].iloc[i] + data['close'].iloc[i]) / 2
        if mid_price != 0:
            range_vol = (data['high'].iloc[i] - data['low'].iloc[i]) / mid_price
        else:
            range_vol = 0
            
        if data['close'].iloc[i-1] != 0:
            gap_vol = abs(data['open'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
        else:
            gap_vol = 0
            
        total_daily_volatility[i] = range_vol + gap_vol
        
        # Volatility Multiplier
        if i >= 10:
            vol_avg = np.mean(total_daily_volatility[i-10:i])
            if total_daily_volatility[i] > 1.5 * vol_avg:
                volatility_multiplier[i] = 1.8
            elif total_daily_volatility[i] < 0.6 * vol_avg:
                volatility_multiplier[i] = 0.4
            else:
                volatility_multiplier[i] = 1.0
    
    # Composite Alpha Construction
    core_signal = np.zeros(len(data))
    final_alpha = np.zeros(len(data))
    
    for i in range(10, len(data)):
        # Core Signal
        gap_momentum_component = gap_momentum[i] * volatility_multiplier[i]
        convergence_component = volume_price_convergence[i] * (1 + abs(total_daily_volatility[i]))
        core_signal[i] = gap_momentum_component * convergence_component
        
        # Final Alpha Factor
        range_ratio = (data['high'].iloc[i] - data['low'].iloc[i]) / data['close'].iloc[i] if data['close'].iloc[i] != 0 else 0
        signal_sign = np.sign(gap_momentum_component + convergence_component)
        final_alpha[i] = core_signal[i] * range_ratio * signal_sign
    
    return pd.Series(final_alpha, index=data.index)
