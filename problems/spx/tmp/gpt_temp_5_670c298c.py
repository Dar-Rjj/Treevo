import pandas as pd
import pandas as pd

def heuristics_v2(df, n=10, m=20):
    # Calculate Intraday High-Low Spread
    intraday_high_low_spread = df['high'] - df['low']
    
    # Calculate Close-Open Spread
    close_open_spread = df['close'] - df['open']
    
    # Calculate Raw Momentum
    raw_momentum = df['close'].diff()
    
    # Adjust Price Momentum by Volume
    weighted_avg_volume = df['volume'].rolling(window=n).mean()
    adjusted_momentum = raw_momentum * (df['volume'] / weighted_avg_volume)
    
    # Incorporate Volume Intensity
    avg_volume_5 = df['volume'].rolling(window=5).mean()
    volume_intensity_factor = 1 / avg_volume_5
    momentum_with_volume_intensity = raw_momentum * volume_intensity_factor
    
    # Detect Volume Spikes
    avg_volume_m = df['volume'].rolling(window=m).mean()
    volume_spike = df['volume'] > 1.5 * avg_volume_m
    adjusted_raw_momentum = raw_momentum.where(~volume_spike, raw_momentum * 0.5)
    
    # Combine Intraday Factors
    intraday_factors = intraday_high_low_spread + close_open_spread
    
    # Compute Next Day Open to Current Day Close
    next_day_open_to_close = df['open'].shift(-1) - df['close']
    
    # Combine for Initial Alpha Signal
    initial_alpha_signal = (intraday_factors + next_day_open_to_close) * df['volume']
    
    # Calculate Daily Price Change
    daily_price_change = df['close'].diff()
    
    # Calculate Volume Change
    volume_change = df['volume'].diff()
    
    # Compute Smoothed Price Momentum
    smoothed_price_momentum = daily_price_change.rolling(window=5).mean()
    
    # Compute Weighted Volume Indicator
    weighted_volume_indicator = (volume_change * volume_change.sign()).rolling(window=5).mean()
    
    # Calculate Weighted Price Movement
    weighted_price_movement = (df['close'] - df['open']) * df['volume'] / avg_volume_5
    
    # Final Alpha Signal
    final_alpha_signal = (
        adjusted_momentum * smoothed_price_momentum * weighted_volume_indicator +
        initial_alpha_signal
    ) * intraday_factors * weighted_price_movement
    
    return final_alpha_signal
