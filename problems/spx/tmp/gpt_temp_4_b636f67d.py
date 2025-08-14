import pandas as pd
import pandas as pd

def heuristics_v2(df, n=20, m=10, k=14):
    # Calculate Daily Price Change
    df['daily_price_change'] = df['close'].diff()
    
    # Calculate Raw Momentum
    df['raw_momentum'] = df['daily_price_change'].rolling(window=n).sum() / n
    
    # Identify Volume Spikes
    df['volume_ma'] = df['volume'].rolling(window=m).mean()
    df['volume_spike'] = (df['volume'] > 1.5 * df['volume_ma']).astype(int)
    
    # Adjust Momentum by Volume Spike
    df['adjusted_momentum'] = df['raw_momentum'] * (1 - df['volume_spike'] * 0.5)
    
    # Calculate Smoothed Price Momentum
    df['smoothed_momentum'] = df['daily_price_change'].rolling(window=5).sum() / 5
    
    # Compute Volume Weight
    df['average_volume'] = df['volume'].rolling(window=n).mean()
    df['normalized_volume'] = df['volume'] / df['average_volume']
    df['volume_weight'] = df['normalized_volume'].rolling(window=n).mean()
    
    # Combine Smoothed Price Momentum and Volume Weight
    df['momentum_volume_weighted'] = df['smoothed_momentum'] * df['volume_weight']
    
    # Calculate Intraday High-Low Spread Factor
    df['intraday_high_low_spread'] = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['close'] - df['open']
    
    # Combine Intraday High-Low Spread Factor and Close-to-Open Return
    df['intraday_factor'] = df['intraday_high_low_spread'] + df['close_to_open_return']
    
    # Calculate Weighted Price Movement
    df['weighted_price_movement'] = (df['close'] - df['open']) * df['volume']
    df['average_volume_5_days'] = df['volume'].rolling(window=5).sum() / 5
    df['normalized_weighted_price_movement'] = df['weighted_price_movement'] / df['average_volume_5_days']
    
    # Introduce Volatility Factor
    df['daily_range'] = df['high'] - df['low']
    df['true_range'] = df[['high', 'low']].apply(lambda x: max(x[0] - x[1], abs(x[0] - df['close'].shift(1)), abs(x[1] - df['close'].shift(1))), axis=1)
    df['atr'] = df['true_range'].rolling(window=k).mean()
    
    # Adjust Final Alpha Factor by Volatility
    df['final_alpha_factor'] = (df['adjusted_momentum'] * df['intraday_factor'] * df['normalized_weighted_price_movement']) / df['atr'] + df['momentum_volume_weighted']
    
    # Introduce Short-Term Reversal Indicator
    df['one_day_price_change'] = df['close'] - df['close'].shift(1)
    df['short_term_reversal'] = df['one_day_price_change'].apply(lambda x: 0.01 if x < 0 else 0)
    
    # Introduce Price Trend Indicator
    df['price_trend'] = df['close'] - df['close'].rolling(window=5).mean()
    
    # Combine All Indicators
    df['alpha_factor'] = df['final_alpha_factor'] + df['short_term_reversal'] + df['price_trend']
    
    return df['alpha_factor'].dropna()

# Example usage:
# alpha_factor = heuristics_v2(df)
