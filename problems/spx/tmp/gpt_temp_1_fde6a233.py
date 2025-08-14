import pandas as pd
import pandas as pd

def heuristics_v2(df, m=5, n=10, volume_spike_threshold=1.5):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate Daily Momentum with Volume Weight
    df['daily_momentum_vol_weight'] = (df['close'] - df['close'].shift(1)) * df['volume']
    
    # Determine Intraday Reversal Signal
    df['intraday_move'] = df['high'] - df['close']
    df['trade_intensity'] = df['volume'] / (df['amount'] / df['volume'])
    
    # Adjust Momentum by Intraday Volatility and Open-Price Gradient
    df['open_price_gradient'] = df['open'] - df['open'].shift(1)
    df['intraday_volatility'] = df['high'] - df['low']
    df['adjusted_momentum'] = df['daily_momentum_vol_weight'] / (df['intraday_volatility'] + df['open_price_gradient']).abs()
    
    # Identify Volume Spikes
    df['avg_volume'] = df['volume'].rolling(window=m).mean()
    df['volume_spike'] = df['volume'] > (df['avg_volume'] * volume_spike_threshold)
    
    # Adjust Momentum by Volume Spike
    df['scaled_momentum'] = df['adjusted_momentum'] * (volume_spike_threshold if df['volume_spike'] else 1)
    
    # Calculate Price Momentum
    df['price_momentum'] = df['close'].pct_change(periods=n)
    
    # Weight Intraday Move by Trade Intensity
    df['weighted_intraday_move'] = df['intraday_move'] * df['trade_intensity']
    
    # Weight Adjusted Daily Momentum by Trade Intensity
    df['average_price'] = (df['high'] + df['low']) / 2
    df['trade_intensity_avg_price'] = df['volume'] / df['average_price']
    df['weighted_adjusted_momentum'] = df['scaled_momentum'] * df['trade_intensity_avg_price']
    
    # Incorporate Intraday Range Expansion
    df['prev_intraday_range'] = df['intraday_range'].shift(1)
    df['range_expansion'] = df['intraday_range'] > df['prev_intraday_range']
    df['weighted_intraday_move_range'] = df['weighted_intraday_move'] * (1.5 if df['range_expansion'] else 0.5)
    
    # Combine All Weighted Components
    df['combined_momentum'] = (df['weighted_intraday_move_range'] + 
                               df['weighted_adjusted_momentum'] + 
                               df['weighted_intraday_move'] + 
                               df['price_momentum'])
    
    # Final Factor Calculation
    df['alpha_factor'] = df['combined_momentum'].apply(lambda x: 1 if x > 0 else 0)
    
    return df['alpha_factor']
