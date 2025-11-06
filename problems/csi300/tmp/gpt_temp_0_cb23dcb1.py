import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Volatility-Normalized Breakout with Volume Convergence alpha factor
    """
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Multi-Timeframe Breakout Detection
    short_breakout = pd.Series(index=df.index, dtype=int)
    medium_breakout = pd.Series(index=df.index, dtype=int)
    long_breakout = pd.Series(index=df.index, dtype=int)
    
    for i in range(len(df)):
        if i >= 20:  # Ensure enough data for all timeframes
            # Short-term breakout (5-day)
            short_high_max = df['high'].iloc[i-5:i].max()
            short_breakout.iloc[i] = 1 if df['high'].iloc[i] > short_high_max else 0
            
            # Medium-term breakout (10-day)
            medium_high_max = df['high'].iloc[i-10:i].max()
            medium_breakout.iloc[i] = 1 if df['high'].iloc[i] > medium_high_max else 0
            
            # Long-term breakout (20-day)
            long_high_max = df['high'].iloc[i-20:i].max()
            long_breakout.iloc[i] = 1 if df['high'].iloc[i] > long_high_max else 0
    
    breakout_score = short_breakout + medium_breakout + long_breakout
    
    # Multi-Timeframe Volatility Normalization
    short_vol = pd.Series(index=df.index, dtype=float)
    medium_vol = pd.Series(index=df.index, dtype=float)
    long_vol = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i >= 20:
            # Short-term volatility (5-day)
            short_high_max = df['high'].iloc[i-5:i+1].max()
            short_low_min = df['low'].iloc[i-5:i+1].min()
            short_vol.iloc[i] = (short_high_max - short_low_min) / df['close'].iloc[i]
            
            # Medium-term volatility (10-day)
            medium_high_max = df['high'].iloc[i-10:i+1].max()
            medium_low_min = df['low'].iloc[i-10:i+1].min()
            medium_vol.iloc[i] = (medium_high_max - medium_low_min) / df['close'].iloc[i]
            
            # Long-term volatility (20-day)
            long_high_max = df['high'].iloc[i-20:i+1].max()
            long_low_min = df['low'].iloc[i-20:i+1].min()
            long_vol.iloc[i] = (long_high_max - long_low_min) / df['close'].iloc[i]
    
    volatility_adjusted_breakout = breakout_score / (1 + short_vol + medium_vol + long_vol)
    
    # Multi-Horizon Volume Convergence
    vol_ratio_5d = pd.Series(index=df.index, dtype=float)
    vol_ratio_10d = pd.Series(index=df.index, dtype=float)
    vol_ratio_20d = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i >= 20:
            # 5-day volume ratio
            vol_5d_mean = df['volume'].iloc[i-5:i].mean()
            vol_ratio_5d.iloc[i] = df['volume'].iloc[i] / vol_5d_mean if vol_5d_mean > 0 else 1
            
            # 10-day volume ratio
            vol_10d_mean = df['volume'].iloc[i-10:i].mean()
            vol_ratio_10d.iloc[i] = df['volume'].iloc[i] / vol_10d_mean if vol_10d_mean > 0 else 1
            
            # 20-day volume ratio
            vol_20d_mean = df['volume'].iloc[i-20:i].mean()
            vol_ratio_20d.iloc[i] = df['volume'].iloc[i] / vol_20d_mean if vol_20d_mean > 0 else 1
    
    volume_convergence = (vol_ratio_5d + vol_ratio_10d + vol_ratio_20d) / 3
    
    # Signal Blending and Enhancement
    volatility_normalized_signal = volatility_adjusted_breakout * volume_convergence
    
    # Price Momentum Confirmation
    price_momentum = pd.Series(index=df.index, dtype=float)
    for i in range(1, len(df)):
        price_momentum.iloc[i] = (df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
    
    enhanced_signal = volatility_normalized_signal * (1 + abs(price_momentum))
    
    # Final Alpha Factor
    alpha = enhanced_signal
    
    return alpha
