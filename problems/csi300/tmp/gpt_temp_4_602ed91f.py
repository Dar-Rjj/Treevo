import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate True Range
    high_low = df['high'] - df['low']
    high_prev_close = abs(df['high'] - df['close'].shift(1))
    low_prev_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    
    # Calculate TR moving averages
    tr_ma_5 = tr.rolling(window=5, min_periods=5).mean()
    tr_ma_10 = tr.rolling(window=10, min_periods=10).mean()
    
    # Measure compression patterns
    compression_flag = tr < tr_ma_10
    compression_duration = compression_flag.astype(int).groupby((~compression_flag).cumsum()).cumsum()
    compression_intensity = (tr_ma_10 - tr) / tr_ma_10
    
    # Initialize output series
    alpha_signal = pd.Series(index=df.index, dtype=float)
    
    for i in range(10, len(df)):
        current_date = df.index[i]
        
        # Find current compression period
        current_comp_duration = compression_duration.iloc[i]
        if current_comp_duration == 0:
            continue
            
        # Get compression period data
        comp_start_idx = i - current_comp_duration + 1
        comp_data = df.iloc[comp_start_idx:i+1]
        
        # Calculate compression period statistics
        highest_high = comp_data['high'].max()
        lowest_low = comp_data['low'].min()
        avg_volume_comp = comp_data['volume'].mean()
        
        # Detect breakouts
        current_close = df['close'].iloc[i]
        current_volume = df['volume'].iloc[i]
        
        upward_breakout = (current_close > highest_high) and (current_volume > 1.5 * avg_volume_comp)
        downward_breakout = (current_close < lowest_low) and (current_volume > 1.5 * avg_volume_comp)
        
        # Calculate breakout strengths
        upward_strength = 0
        downward_strength = 0
        
        if upward_breakout:
            upward_strength = (current_close - highest_high) / highest_high * current_comp_duration
        
        if downward_breakout:
            downward_strength = (lowest_low - current_close) / lowest_low * current_comp_duration
        
        # Calculate breakout asymmetry ratio
        if upward_strength + downward_strength > 0:
            breakout_asymmetry = upward_strength / (upward_strength + downward_strength + 1e-8)
        else:
            breakout_asymmetry = 0.5
        
        # Analyze volume distribution during compression
        up_days = comp_data[comp_data['close'] > comp_data['open']]
        down_days = comp_data[comp_data['close'] < comp_data['open']]
        
        up_volume = up_days['volume'].sum() if len(up_days) > 0 else 0
        down_volume = down_days['volume'].sum() if len(down_days) > 0 else 0
        total_volume_comp = comp_data['volume'].sum()
        
        if total_volume_comp > 0:
            volume_skew = up_volume / total_volume_comp
        else:
            volume_skew = 0.5
        
        # Generate final alpha signal
        compression_factor = np.log1p(current_comp_duration)
        
        if upward_breakout:
            # Positive bias for upward breakouts (favor continuation)
            signal = breakout_asymmetry * volume_skew * compression_factor
        elif downward_breakout:
            # Negative bias for downward breakouts (favor reversal)
            signal = -(1 - breakout_asymmetry) * (1 - volume_skew) * compression_factor
        else:
            signal = 0
        
        alpha_signal.iloc[i] = signal
    
    # Fill NaN values with 0
    alpha_signal = alpha_signal.fillna(0)
    
    return alpha_signal
