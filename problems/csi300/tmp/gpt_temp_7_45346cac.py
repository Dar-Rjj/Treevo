import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate required periods
    for i in range(20, len(df)):
        current_data = df.iloc[i]
        prev_data = df.iloc[i-1] if i > 0 else None
        
        # Intraday Momentum Analysis
        intraday_return = (current_data['close'] - current_data['open']) / current_data['open']
        prev_momentum = (df.iloc[i-1]['close'] - df.iloc[i-1]['open']) / df.iloc[i-1]['open'] if i > 0 else 0
        momentum_continuation = 1 if np.sign(intraday_return) == np.sign(prev_momentum) else 0
        
        # Breakout Event Detection
        high_20 = df['high'].iloc[i-20:i].max()
        low_20 = df['low'].iloc[i-20:i].min()
        
        high_breakout = current_data['high'] > high_20
        low_breakout = current_data['low'] < low_20
        
        high_breakout_mag = (current_data['high'] - high_20) / high_20 if high_breakout else 0
        low_breakout_mag = (low_20 - current_data['low']) / low_20 if low_breakout else 0
        
        breakout_direction = 1 if high_breakout else (-1 if low_breakout else 0)
        breakout_magnitude = high_breakout_mag if high_breakout else low_breakout_mag
        
        # Momentum Persistence Strength
        momentum_magnitude = abs(intraday_return) * abs(prev_momentum)
        
        # Multi-day Momentum Alignment
        if i >= 5:
            return_3d = (current_data['close'] - df.iloc[i-3]['close']) / df.iloc[i-3]['close']
            return_5d = (current_data['close'] - df.iloc[i-5]['close']) / df.iloc[i-5]['close']
            
            sign_intraday = np.sign(intraday_return)
            sign_3d = np.sign(return_3d)
            sign_5d = np.sign(return_5d)
            
            multi_day_alignment = 1 if (sign_intraday == sign_3d == sign_5d) else 0
        else:
            multi_day_alignment = 0
            
        persistence_score = momentum_magnitude * multi_day_alignment
        
        # Volume Confirmation
        volume_20_mean = df['volume'].iloc[i-20:i].mean()
        volume_5_mean = df['volume'].iloc[i-5:i].mean() if i >= 5 else volume_20_mean
        
        volume_spike = 1 if current_data['volume'] > 1.5 * volume_20_mean else 0
        volume_trend = 1 if current_data['volume'] > volume_5_mean else 0
        volume_strength = volume_spike * volume_trend
        
        # Price Range Analysis
        current_range = current_data['high'] - current_data['low']
        avg_range_20 = (df['high'].iloc[i-20:i] - df['low'].iloc[i-20:i]).mean()
        range_expansion = current_range / avg_range_20 if avg_range_20 > 0 else 1
        
        # Factor Combination
        core_breakout_factor = breakout_direction * breakout_magnitude * persistence_score
        volume_enhanced_factor = core_breakout_factor * volume_strength
        range_adjusted_factor = volume_enhanced_factor * range_expansion
        final_factor = range_adjusted_factor * momentum_continuation
        
        result.iloc[i] = final_factor
    
    # Fill NaN values with 0
    result = result.fillna(0)
    
    return result
