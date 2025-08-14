import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Simple Momentum
    n = 10  # Example: 10-day momentum
    simple_momentum = df['close'] / df['close'].shift(n) - 1
    
    # Volume Adjusted Component
    daily_volume_change = df['volume'] / df['volume'].shift(1)
    volume_adjusted_momentum = simple_momentum * daily_volume_change
    
    # Enhanced Price Reversal Sensitivity
    high_low_spread = df['high'] - df['low']
    open_close_spread = (df['open'] - df['close']).abs()
    weighted_high_low_spread = high_low_spread * df['volume']
    weighted_open_close_spread = open_close_spread * df['volume']
    combined_weighted_spread = weighted_high_low_spread + weighted_open_close_spread
    
    # Incorporate Enhanced Price Gaps
    gap_oc = df['open'] - df['close'].shift(1)
    gap_hl = df['high'] - df['low']
    combined_gaps = volume_adjusted_momentum + gap_oc + gap_hl
    
    # Introduce a Smoothing Factor for High-Low Spread
    smoothed_high_low_spread = high_low_spread.ewm(span=5).mean()
    
    # Combine Momentum and Close-to-Low Distance
    close_to_low_distance = df['close'] - df['low']
    momentum_close_to_low = simple_momentum * close_to_low_distance
    
    # Measure Volume Impact
    vol_ema_10 = df['volume'].ewm(span=10).mean()
    
    # Confirm with Volume
    vol_ma_5 = df['volume'].rolling(window=5).mean()
    vol_ma_20 = df['volume'].rolling(window=20).mean()
    confirmed_momentum = combined_gaps
    confirmed_momentum = confirmed_monoment.where(vol_ma_5 > vol_ma_20, confirmed_momentum * 0.5)
    
    # Adjust Momentum by ATR
    true_range = df[['high', 'low']].sub(df['close'].shift(1), axis=0).abs().max(axis=1)
    atr = true_range.rolling(window=14).mean()
    adj_momentum_atr = confirmed_momentum / atr
    
    # Final Combination
    final_combination = (volume_adjusted_momentum - combined_weighted_spread) / vol_ema_10
    
    # Final Volume Adjustment
    vol_ma_n = df['volume'].rolling(window=14).mean()
    final_factor = adj_momentum_atr * vol_ma_n
    
    return final_factor
