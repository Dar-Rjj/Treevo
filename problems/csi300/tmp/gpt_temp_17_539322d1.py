import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate rolling windows
    df['high_roll_5'] = df['high'].rolling(window=5, min_periods=1).mean()
    df['low_roll_5'] = df['low'].rolling(window=5, min_periods=1).mean()
    df['volume_roll_5'] = df['volume'].rolling(window=5, min_periods=1).mean()
    
    # Fractal Acceleration Framework
    upside_fractal_acceleration = pd.Series(index=df.index, dtype=float)
    downside_fractal_acceleration = pd.Series(index=df.index, dtype=float)
    
    for i in range(5, len(df)):
        # Upside fractal acceleration
        if df['high'].iloc[i] > df['high'].iloc[i-1] and df['low'].iloc[i] > df['low'].iloc[i-1]:
            current_accel = df['high'].iloc[i] - df['low'].iloc[i-1]
            avg_accel = (df['high'].iloc[i-4:i+1] - df['low'].iloc[i-5:i]).mean()
            upside_fractal_acceleration.iloc[i] = current_accel - avg_accel
        else:
            upside_fractal_acceleration.iloc[i] = 0
        
        # Downside fractal acceleration
        if df['high'].iloc[i] < df['high'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i-1]:
            current_accel = df['high'].iloc[i-1] - df['low'].iloc[i]
            avg_accel = (df['high'].iloc[i-5:i] - df['low'].iloc[i-4:i+1]).mean()
            downside_fractal_acceleration.iloc[i] = current_accel - avg_accel
        else:
            downside_fractal_acceleration.iloc[i] = 0
    
    net_fractal_acceleration = upside_fractal_acceleration - downside_fractal_acceleration
    
    # Volume-Amount Synchronization
    volume_fractal_intensity = pd.Series(index=df.index, dtype=float)
    participation_score = pd.Series(index=df.index, dtype=float)
    volume_amount_alignment = pd.Series(index=df.index, dtype=float)
    
    for i in range(5, len(df)):
        # Volume fractal intensity
        vol_ratio_current = df['volume'].iloc[i] / df['volume_roll_5'].iloc[i]
        vol_ratio_2day = (df['volume'].iloc[i] / df['volume'].iloc[i-2] - 1) if df['volume'].iloc[i-2] != 0 else 0
        vol_ratio_5day = (df['volume'].iloc[i] / df['volume'].iloc[i-5] - 1) if df['volume'].iloc[i-5] != 0 else 0
        
        if vol_ratio_5day != 0:
            volume_fractal_intensity.iloc[i] = vol_ratio_current * (vol_ratio_2day / vol_ratio_5day)
        else:
            volume_fractal_intensity.iloc[i] = vol_ratio_current
        
        # Participation score
        price_range = df['high'].iloc[i] - df['low'].iloc[i]
        if price_range != 0:
            amount_density = df['amount'].iloc[i] / price_range
            amount_change = (df['amount'].iloc[i] / df['amount'].iloc[i-1] - 1) if df['amount'].iloc[i-1] != 0 else 0
            participation_score.iloc[i] = amount_density * amount_change
        else:
            participation_score.iloc[i] = 0
        
        # Volume-amount alignment
        vol_sign = np.sign(df['volume'].iloc[i] - df['volume'].iloc[i-1]) if i > 0 else 0
        amount_sign = np.sign(df['amount'].iloc[i] - df['amount'].iloc[i-1]) if i > 0 else 0
        volume_amount_alignment.iloc[i] = vol_sign * amount_sign
    
    # Multi-Timeframe Decay Analysis
    fractal_momentum_divergence = pd.Series(index=df.index, dtype=float)
    volume_anchored_decay = pd.Series(index=df.index, dtype=float)
    
    net_fractal_roll_5 = net_fractal_acceleration.rolling(window=5, min_periods=1).mean()
    
    for i in range(len(df)):
        fractal_momentum_divergence.iloc[i] = net_fractal_acceleration.iloc[i] - net_fractal_roll_5.iloc[i]
        
        if df['volume_roll_5'].iloc[i] != 0:
            volume_ratio = df['volume'].iloc[i] / df['volume_roll_5'].iloc[i]
            volume_anchored_decay.iloc[i] = (fractal_momentum_divergence.iloc[i] * net_fractal_roll_5.iloc[i]) * volume_ratio
        else:
            volume_anchored_decay.iloc[i] = 0
    
    # Intraday Absorption Dynamics
    price_efficiency = pd.Series(index=df.index, dtype=float)
    overnight_gap = pd.Series(index=df.index, dtype=float)
    fractal_gap_absorption = pd.Series(index=df.index, dtype=float)
    
    for i in range(1, len(df)):
        # Price efficiency
        price_range = df['high'].iloc[i] - df['low'].iloc[i]
        if price_range != 0:
            price_efficiency.iloc[i] = (df['close'].iloc[i] - df['open'].iloc[i]) / price_range
        else:
            price_efficiency.iloc[i] = 0
        
        # Overnight gap
        if df['close'].iloc[i-1] != 0:
            overnight_gap.iloc[i] = df['open'].iloc[i] / df['close'].iloc[i-1] - 1
        else:
            overnight_gap.iloc[i] = 0
        
        # Fractal gap absorption
        fractal_gap_absorption.iloc[i] = overnight_gap.iloc[i] * price_efficiency.iloc[i] * net_fractal_acceleration.iloc[i]
    
    # Asymmetric Fractal-Volume Integration
    fractal_momentum_asymmetry = pd.Series(index=df.index, dtype=float)
    high_intensity_fractal = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        price_range = df['high'].iloc[i] - df['low'].iloc[i]
        if price_range != 0:
            upside_weight = (df['close'].iloc[i] - df['low'].iloc[i]) / price_range
            downside_weight = (df['high'].iloc[i] - df['close'].iloc[i]) / price_range
            
            numerator = upside_weight * upside_fractal_acceleration.iloc[i] - downside_weight * downside_fractal_acceleration.iloc[i]
            denominator = abs(upside_weight * upside_fractal_acceleration.iloc[i]) + abs(downside_weight * downside_fractal_acceleration.iloc[i])
            
            if denominator != 0:
                fractal_momentum_asymmetry.iloc[i] = numerator / denominator
            else:
                fractal_momentum_asymmetry.iloc[i] = 0
        else:
            fractal_momentum_asymmetry.iloc[i] = 0
        
        # High intensity fractal
        high_intensity_fractal.iloc[i] = net_fractal_acceleration.iloc[i] * volume_fractal_intensity.iloc[i]
    
    # Composite Alpha Factor
    for i in range(len(df)):
        # Core fractal signal
        core_fractal_signal = net_fractal_acceleration.iloc[i] * volume_fractal_intensity.iloc[i] * participation_score.iloc[i]
        
        # Decay confirmation
        decay_confirmation = volume_anchored_decay.iloc[i] * volume_amount_alignment.iloc[i]
        
        # Absorption enhancement
        absorption_enhancement = fractal_gap_absorption.iloc[i] * fractal_momentum_asymmetry.iloc[i]
        
        # Final composite factor
        result.iloc[i] = core_fractal_signal + decay_confirmation + absorption_enhancement
    
    return result
