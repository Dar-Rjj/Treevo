import pandas as pd
import pandas as pd

def heuristics_v2(df, n=10, m=20, k=14):
    # Calculate Daily Price Change
    df['daily_price_change'] = df['close'] - df['close'].shift(1)
    
    # Calculate Raw Momentum
    df['raw_momentum'] = df['daily_price_change'].rolling(window=n).sum() / n
    
    # Identify Volume Spikes
    df['volume_ma'] = df['volume'].rolling(window=m).mean()
    df['volume_spike'] = (df['volume'] > 1.5 * df['volume_ma']).astype(int)
    
    # Adjust Momentum by Volume Spike
    df['adjusted_momentum'] = df.apply(
        lambda row: row['raw_momentum'] * 0.7 if row['volume_spike'] == 1 else row['raw_momentum'], axis=1
    )
    
    # Calculate Smoothed Price Momentum
    df['smoothed_price_momentum'] = df['daily_price_change'].rolling(window=5).sum() / 5
    
    # Compute Volume Weight
    df['avg_volume'] = df['volume'].rolling(window=n).mean()
    df['norm_volume'] = df['volume'] / df['avg_volume']
    df['volume_weight'] = df['norm_volume'].rolling(window=n).mean()
    
    # Combine Smoothed Price Momentum and Volume Weight
    df['combined_momentum'] = df['smoothed_price_momentum'] * df['volume_weight']
    
    # Calculate Intraday High-Low Spread Factor
    df['intraday_high_low_spread'] = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['close'] - df['open']
    
    # Combine Intraday High-Low Spread Factor and Close-to-Open Return
    df['combined_intraday_factor'] = df['intraday_high_low_spread'] + df['close_to_open_return']
    
    # Calculate Weighted Price Movement
    df['weighted_price_movement'] = (df['close'] - df['open']) * df['volume']
    df['avg_5_day_volume'] = df['volume'].rolling(window=5).mean()
    df['normalized_weighted_price_movement'] = df['weighted_price_movement'] / df['avg_5_day_volume']
    
    # Introduce Volatility Factor
    df['daily_range'] = df['high'] - df['low']
    df['true_range'] = df[['daily_range', (df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()]].max(axis=1)
    df['atr'] = df['true_range'].rolling(window=k).mean()
    
    # Adjust Final Alpha Factor by Volatility
    df['final_alpha_factor'] = (
        df['adjusted_momentum'] * df['combined_intraday_factor'] * 
        df['normalized_weighted_price_movement'] * (1 / df['atr'])
    ) + df['adjusted_momentum']
    
    return df['final_alpha_factor']
