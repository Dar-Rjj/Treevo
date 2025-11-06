import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(5, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # Volatility Regime Classification
        # Short-term Volatility
        short_term_vol = []
        for j in range(max(0, i-4), i+1):
            if j > 0:
                price_range = current_data.iloc[j]['high'] - current_data.iloc[j]['low']
                price_change = abs(current_data.iloc[j]['close'] - current_data.iloc[j-1]['close'])
                if price_change > 0:
                    short_term_vol.append(price_range / price_change)
        short_term_vol_avg = np.mean(short_term_vol) if short_term_vol else 1.0
        
        # Medium-term Volatility
        if i >= 5:
            current_range = current_data.iloc[i]['high'] - current_data.iloc[i]['low']
            past_range = current_data.iloc[i-5]['high'] - current_data.iloc[i-5]['low']
            medium_term_vol = current_range / past_range if past_range > 0 else 1.0
        else:
            medium_term_vol = 1.0
        
        regime_type = np.sign(short_term_vol_avg - medium_term_vol)
        
        # Multi-Scale Rejection Asymmetry
        # Intraday Rejection Asymmetry
        high_t = current_data.iloc[i]['high']
        low_t = current_data.iloc[i]['low']
        open_t = current_data.iloc[i]['open']
        close_t = current_data.iloc[i]['close']
        
        upper_shadow_eff = (high_t - max(open_t, close_t)) / (high_t - low_t) if (high_t - low_t) > 0 else 0
        lower_shadow_eff = (min(open_t, close_t) - low_t) / (high_t - low_t) if (high_t - low_t) > 0 else 0
        intraday_rej_asym = (upper_shadow_eff - lower_shadow_eff) * np.sign(close_t - open_t)
        
        # Multi-Timeframe Rejection Momentum
        if i >= 2:
            close_t2 = current_data.iloc[i-2]['close']
            close_t1 = current_data.iloc[i-1]['close']
            three_day_high_rej = (high_t - max(close_t2, close_t1, close_t)) / (high_t - low_t) if (high_t - low_t) > 0 else 0
            three_day_low_rej = (min(close_t2, close_t1, close_t) - low_t) / (high_t - low_t) if (high_t - low_t) > 0 else 0
            multi_timeframe_rej_mom = (three_day_high_rej - three_day_low_rej) * np.sign(close_t - open_t)
        else:
            multi_timeframe_rej_mom = 0
        
        rejection_velocity_conv = intraday_rej_asym * multi_timeframe_rej_mom
        
        # Regime-Adaptive Volume Dynamics
        if i > 0:
            volume_t = current_data.iloc[i]['volume']
            volume_t1 = current_data.iloc[i-1]['volume']
            volume_momentum = (volume_t / volume_t1 - 1) * np.sign(close_t - open_t) if volume_t1 > 0 else 0
        else:
            volume_momentum = 0
        
        if i >= 3:
            volume_t3 = current_data.iloc[i-3]['volume']
            volume_acceleration = (volume_t / volume_t3) ** (1/3) - 1 if volume_t3 > 0 else 0
        else:
            volume_acceleration = 0
        
        regime_volume_conf = volume_momentum * volume_acceleration
        
        # Price Efficiency Patterns
        if close_t > open_t:
            up_move_efficiency = (close_t - open_t) / (high_t - low_t) if (high_t - low_t) > 0 else 0
            down_move_inefficiency = 0
        else:
            up_move_efficiency = 0
            down_move_inefficiency = (open_t - close_t) / (high_t - low_t) if (high_t - low_t) > 0 else 0
        
        if i > 0:
            high_t1 = current_data.iloc[i-1]['high']
            low_t1 = current_data.iloc[i-1]['low']
            open_t1 = current_data.iloc[i-1]['open']
            close_t1 = current_data.iloc[i-1]['close']
            
            current_efficiency = abs(close_t - open_t) / (high_t - low_t) if (high_t - low_t) > 0 else 0
            prev_efficiency = abs(close_t1 - open_t1) / (high_t1 - low_t1) if (high_t1 - low_t1) > 0 else 0
            efficiency_velocity_conv = volume_momentum * (current_efficiency - prev_efficiency)
        else:
            efficiency_velocity_conv = 0
        
        # Composite Alpha Construction
        core_signal = rejection_velocity_conv * regime_volume_conf
        enhancement = efficiency_velocity_conv * down_move_inefficiency
        final_alpha = regime_type * core_signal * enhancement
        
        alpha.iloc[i] = final_alpha
    
    # Fill initial NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
