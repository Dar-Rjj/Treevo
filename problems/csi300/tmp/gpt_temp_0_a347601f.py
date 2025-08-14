import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Simple Momentum
    simple_momentum = df['close'].pct_change(periods=10)
    
    # Volume Adjusted Component
    daily_volume_change = df['volume'].pct_change()
    volume_adjusted_component = daily_volume_change * simple_momentum
    
    # Enhanced Price Reversal Sensitivity
    high_low_spread = df['high'] - df['low']
    open_close_spread = (df['open'] - df['close']).abs()
    weighted_high_low_spread = high_low_spread * df['volume']
    weighted_open_close_spread = open_close_spread * df['volume']
    combined_weighted_spreads = weighted_high_low_spread + weighted_open_close_spread
    smoothed_high_low_spread = high_low_spread.ewm(span=5).mean()
    
    # Combine Momentum and Close-to-Low Distance
    close_to_low_distance = df['close'] - df['low']
    combined_momentum = simple_momentum * close_to_low_distance
    
    # Measure Volume Impact
    volume_ema_10 = df['volume'].ewm(span=10).mean()
    
    # Confirm with Volume
    volume_ema_5 = df['volume'].ewm(span=5).mean()
    volume_ema_20 = df['volume'].ewm(span=20).mean()
    volume_confirmation = volume_ema_5 > volume_ema_20
    combined_momentum_confirmed = combined_momentum.where(volume_confirmation, combined_momentum * 0.8)
    combined_momentum_confirmed = combined_momentum_confirmed.where(~volume_confirmation, combined_momentum * 1.2)
    
    # Adjust Momentum by ATR
    true_range = df[['high', 'low', 'close']].apply(lambda x: max(x[0] - x[1], abs(x[0] - x[2]), abs(x[1] - x[2])), axis=1)
    atr = true_range.ewm(span=14).mean()
    adjusted_combined_momentum = combined_momentum_confirmed / atr
    volume_ema_n = df['volume'].ewm(span=14).mean()
    adjusted_combined_momentum = adjusted_combined_momentum * volume_ema_n
    
    # Final Combination
    final_alpha = (volume_adjusted_component 
                   - combined_weighted_spreads 
                   + adjusted_combined_momentum) / volume_ema_10
    
    return final_alpha
