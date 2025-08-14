import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Compute Price Change
    df['price_change'] = df['close'].diff()
    
    # Incorporate Volume Impact Factor
    df['volume_impact'] = df['volume'] * df['price_change'].abs()
    
    # Integrate Historical High-Low Range and Momentum Contributions
    df['high_low_range_5d_sum'] = df['high_low_range'].rolling(window=5).sum()
    df['momentum_contribution'] = df['price_change'] * df['high_low_range']
    df['weighted_momentum'] = df['momentum_contribution'].where(df['price_change'] > 0, 0)
    df['accumulated_weighted_momentum'] = df['weighted_momentum'].rolling(window=5).sum()
    
    # Adjust for Market Sentiment
    df['volatility'] = (df['high'] - df['low']) / df['close']
    df['avg_volatility_5d'] = df['volatility'].rolling(window=5).mean()
    threshold = df['avg_volatility_5d'].shift(1)  # Use the previous day's average volatility as the threshold
    
    def adjust_factor(row):
        if row['accumulated_weighted_momentum'] > row['threshold']:
            return row['accumulated_weighted_momentum'] * 1.5  # Increase factor
        else:
            return row['accumulated_weighted_momentum'] * 0.5  # Decrease factor
    
    df['adjusted_factor'] = df.apply(lambda row: adjust_factor(row), axis=1)
    
    # Finalize and Output Alpha Factor
    alpha_factor = df['adjusted_factor']
    return alpha_factor
