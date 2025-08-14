import pandas as pd
import pandas as pd

def heuristics_v2(df, N=10):
    # Calculate Daily Price Movement
    daily_price_movement = df['close'] - df['open']
    
    # Adjust for Intraday Volatility
    intraday_volatility = df['high'] - df['low']
    
    # Calculate Volume-Weighted Price Movement
    volume_weighted_price_movement = (daily_price_movement * df['volume']) / intraday_volatility
    
    # Incorporate Transaction Amount Impact
    transaction_amount_impact = (df['amount'] * volume_weighted_price_movement) / df['volume']
    
    # Identify Directional Days
    df['direction'] = 'Up'
    df.loc[df['close'] < df['open'], 'direction'] = 'Down'
    
    # Count Number of Up and Down Days in Last N Days
    up_days = df['direction'].rolling(window=N).apply(lambda x: (x == 'Up').sum(), raw=False)
    down_days = df['direction'].rolling(window=N).apply(lambda x: (x == 'Down').sum(), raw=False)
    
    # Weight by Volume and Amount
    volume_weighted_directional_counts = (up_days - down_days) * (df['volume'] + df['amount'])
    
    # Combine Volume-Weighted Price Movement and Directional Counts
    combined_volume_weighted = volume_weighted_price_movement + volume_weighted_directional_counts
    
    # Calculate Close Price Momentum
    close_momentum = df['close'] - df['close'].shift(N)
    
    # Calculate Short-Term Price Momentum
    short_term_momentum = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Calculate Long-Term Price Momentum
    long_term_momentum = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    
    # Calculate Volume Trend
    volume_trend = df['volume'].rolling(window=5).mean()
    
    # Calculate Trading Activity Indicator
    trading_activity_indicator = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Adjust for Price Volatility
    atr = df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()
    adjusted_trading_activity = trading_activity_indicator / atr
    
    # Weighted Combination
    weighted_combination = short_term_momentum * volume_trend + long_term_momentum
    
    # Final Alpha Factor
    final_alpha_factor = (combined_volume_weighted + close_momentum + 
                          volume_weighted_directional_counts + adjusted_trading_activity + 
                          weighted_combination)
    
    return final_alpha_factor
