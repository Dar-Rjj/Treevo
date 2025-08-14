import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df, n=10, short_period=10, medium_period=30, long_period=60):
    # Sub-Thought: Calculate Price Momentum
    df['price_momentum'] = df['close'].diff(n)
    
    # Sub-Thought: Identify Reversal Opportunities
    df['daily_return'] = df['close'].pct_change()
    df['n_day_avg_return'] = df['daily_return'].rolling(window=n).mean()
    df['reversal_opportunity'] = df['daily_return'] - df['n_day_avg_return']
    
    # Sub-Thought: Trend Strength Analysis
    df['short_ma'] = df['close'].rolling(window=short_period).mean()
    df['medium_ma'] = df['close'].rolling(window=medium_period).mean()
    df['long_ma'] = df['close'].rolling(window=long_period).mean()
    df['trend_strength'] = (df['short_ma'] - df['long_ma']) / df['long_ma']
    
    # Sub-Thought: Analyze Volume-Price Relationship
    df['obv'] = (df['close'].diff() > 0).astype(int) * df['volume']
    df['obv'] = df['obv'].cumsum()
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Sub-Thought: Investigate Volatility and Range
    df['daily_range'] = df['high'] - df['low']
    df['volatility'] = df['daily_return'].rolling(window=n).std()
    df['high_low_ratio'] = df['high'] / df['low']
    
    # Sub-Thought: Study Market Sentiment and Support/Resistance
    df['support'] = df['low'].rolling(window=n).min()
    df['resistance'] = df['high'].rolling(window=n).max()
    df['rsi'] = 100 - (100 / (1 + (df['daily_return'][df['daily_return'] > 0].rolling(window=n).mean() / 
                                -df['daily_return'][df['daily_return'] < 0].rolling(window=n).mean())))
    df['open_close_gap'] = df['open'] - df['close']
    
    # Combine all factors into a single alpha factor
    df['alpha_factor'] = (df['price_momentum'] + df['reversal_opportunity'] + df['trend_strength'] + 
                          df['obv'] + df['vwap'] + df['volatility'] + df['high_low_ratio'] + 
                          df['rsi'] + df['open_close_gap'])
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    return df['alpha_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
