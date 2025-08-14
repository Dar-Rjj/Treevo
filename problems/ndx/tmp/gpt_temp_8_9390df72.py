import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Return
    df['daily_return'] = df['close'] - df['close'].shift(1)
    
    # Aggregate Daily Return
    df['volume_weighted_return'] = df['daily_return'] * df['volume']
    df['agg_daily_return'] = df['volume_weighted_return'].rolling(window=20).sum()
    
    # Calculate Long-Term Trend Strength
    df['long_term_trend'] = df['close'] - df['close'].shift(60)
    
    # Combine Momentum and Trend Factors
    df['momentum_trend'] = df['agg_daily_return'] + df['long_term_trend']
    
    # Adjust for Volume Trend
    df['volume_change'] = df['volume'] - df['volume'].shift(20)
    df['adjusted_momentum_trend'] = df['momentum_trend'] * (1.5 if df['volume_change'] > 0 else 1/1.5)
    
    # Calculate Intraday Price Movement Ratio
    df['high_low_ratio'] = df['high'] / df['low']
    df['open_close_ratio'] = df['open'] / df['close']
    df['intraday_movement_ratio'] = (df['high_low_ratio'] + df['open_close_ratio']) / 2
    
    # Measure Price-Volatility Alignment
    df['price_range'] = df['high'] - df['low']
    df['weighted_volatility'] = df['price_range'] * df['volume']
    df['alignment'] = df['intraday_movement_ratio'] / df['weighted_volatility']
    
    # Determine Sentiment
    df['sentiment'] = df['alignment'].apply(lambda x: 1 if x > 0 else -1)
    
    # Final Alpha Factor
    df['alpha_factor'] = df['adjusted_momentum_trend'] + df['sentiment']
    
    return df['alpha_factor']
