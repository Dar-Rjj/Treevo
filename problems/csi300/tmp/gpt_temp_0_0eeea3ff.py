import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate price changes
    price_change = df['close'].diff().abs()
    
    # Fractal Momentum Efficiency
    momentum_3d = pd.Series(index=df.index, dtype=float)
    momentum_8d = pd.Series(index=df.index, dtype=float)
    momentum_21d = pd.Series(index=df.index, dtype=float)
    
    # Volume-Fractal Alignment
    volume_align_3d = pd.Series(index=df.index, dtype=float)
    volume_align_8d = pd.Series(index=df.index, dtype=float)
    volume_align_21d = pd.Series(index=df.index, dtype=float)
    
    # Breakout Strength
    breakout_3d = pd.Series(index=df.index, dtype=float)
    breakout_8d = pd.Series(index=df.index, dtype=float)
    breakout_21d = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 21:
            continue
            
        # 3-day calculations
        if i >= 3:
            # Fractal Momentum Efficiency (3-day)
            price_changes_3d = price_change.iloc[i-2:i+1].sum()
            if price_changes_3d != 0:
                momentum_3d.iloc[i] = (df['close'].iloc[i] - df['close'].iloc[i-3]) / price_changes_3d
            else:
                momentum_3d.iloc[i] = 0
            
            # Volume-Fractal Alignment (3-day)
            volume_weighted = (df['volume'].iloc[i-2:i+1] * price_change.iloc[i-2:i+1]).sum()
            if price_changes_3d != 0:
                volume_align_3d.iloc[i] = volume_weighted / price_changes_3d
            else:
                volume_align_3d.iloc[i] = 0
            
            # Breakout Strength (3-day)
            high_max_3d = df['high'].iloc[i-2:i].max()
            low_min_3d = df['low'].iloc[i-2:i].min()
            breakout_3d.iloc[i] = 1 if (df['close'].iloc[i] > high_max_3d or df['close'].iloc[i] < low_min_3d) else 0
        
        # 8-day calculations
        if i >= 8:
            # Fractal Momentum Efficiency (8-day)
            price_changes_8d = price_change.iloc[i-7:i+1].sum()
            if price_changes_8d != 0:
                momentum_8d.iloc[i] = (df['close'].iloc[i] - df['close'].iloc[i-8]) / price_changes_8d
            else:
                momentum_8d.iloc[i] = 0
            
            # Volume-Fractal Alignment (8-day)
            volume_weighted_8d = (df['volume'].iloc[i-7:i+1] * price_change.iloc[i-7:i+1]).sum()
            if price_changes_8d != 0:
                volume_align_8d.iloc[i] = volume_weighted_8d / price_changes_8d
            else:
                volume_align_8d.iloc[i] = 0
            
            # Breakout Strength (8-day)
            high_max_8d = df['high'].iloc[i-7:i].max()
            low_min_8d = df['low'].iloc[i-7:i].min()
            breakout_8d.iloc[i] = 1 if (df['close'].iloc[i] > high_max_8d or df['close'].iloc[i] < low_min_8d) else 0
        
        # 21-day calculations
        # Fractal Momentum Efficiency (21-day)
        price_changes_21d = price_change.iloc[i-20:i+1].sum()
        if price_changes_21d != 0:
            momentum_21d.iloc[i] = (df['close'].iloc[i] - df['close'].iloc[i-21]) / price_changes_21d
        else:
            momentum_21d.iloc[i] = 0
        
        # Volume-Fractal Alignment (21-day)
        volume_weighted_21d = (df['volume'].iloc[i-20:i+1] * price_change.iloc[i-20:i+1]).sum()
        if price_changes_21d != 0:
            volume_align_21d.iloc[i] = volume_weighted_21d / price_changes_21d
        else:
            volume_align_21d.iloc[i] = 0
        
        # Breakout Strength (21-day)
        high_max_21d = df['high'].iloc[i-20:i].max()
        low_min_21d = df['low'].iloc[i-20:i].min()
        breakout_21d.iloc[i] = 1 if (df['close'].iloc[i] > high_max_21d or df['close'].iloc[i] < low_min_21d) else 0
        
        # Alpha Synthesis - Cross-timeframe divergence detection and volume-confirmed momentum
        if i >= 21:
            # Calculate momentum divergence across timeframes
            momentum_divergence = (
                np.sign(momentum_3d.iloc[i]) * np.sign(momentum_8d.iloc[i]) * np.sign(momentum_21d.iloc[i])
            )
            
            # Calculate volume alignment strength
            volume_strength = (
                volume_align_3d.iloc[i] * momentum_3d.iloc[i] +
                volume_align_8d.iloc[i] * momentum_8d.iloc[i] +
                volume_align_21d.iloc[i] * momentum_21d.iloc[i]
            ) / 3.0
            
            # Combine breakout signals with momentum and volume confirmation
            breakout_composite = (
                breakout_3d.iloc[i] * momentum_3d.iloc[i] * volume_align_3d.iloc[i] +
                breakout_8d.iloc[i] * momentum_8d.iloc[i] * volume_align_8d.iloc[i] +
                breakout_21d.iloc[i] * momentum_21d.iloc[i] * volume_align_21d.iloc[i]
            )
            
            # Final alpha factor
            result.iloc[i] = (
                momentum_divergence * 0.3 +
                volume_strength * 0.4 +
                breakout_composite * 0.3
            )
    
    return result
