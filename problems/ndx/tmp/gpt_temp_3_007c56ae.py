import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, N=10, smoothing_factor=0.3):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Adjust Price Change by Intraday Range
    df['adjusted_price_change'] = (df['close'] - df['close'].shift(1)) / df['intraday_range']
    
    # Detect Volume Spikes
    avg_volume = df['volume'].rolling(window=N).mean()
    df['volume_spike'] = df['volume'] > 2 * avg_volume
    
    # Weight by Volume
    df['weighted_adjusted_price_change'] = df['volume'] * df['adjusted_price_change']
    
    # Enhance Momentum on Volume Spike Days
    df['enhanced_momentum'] = df['weighted_adjusted_price_change'] * (2 if df['volume_spike'] else 1)
    
    # Cumulative Momentum
    df['cumulative_momentum'] = df['enhanced_momentum'].rolling(window=N).sum()
    
    # Smoothing Process
    df['smoothed_cumulative_momentum'] = df['cumulative_momentum'].ewm(alpha=smoothing_factor).mean()
    
    # Generate Complexity Score
    momentum_10 = df['smoothed_cumulative_momentum'].rolling(window=10).mean()
    momentum_30 = df['smoothed_cumulative_momentum'].rolling(window=30).mean()
    momentum_60 = df['smoothed_cumulative_momentum'].rolling(window=60).mean()
    std_20 = df['close'].rolling(window=20).std()
    complexity_score = (momentum_10 + momentum_30 + momentum_60) / std_20
    df['complexity_score'] = complexity_score * df['volume'].rolling(window=20).mean()
    
    # Construct Momentum Oscillator
    df['ema_high'] = df['high'].ewm(span=5, adjust=False).mean()
    df['ema_close'] = df['close'].ewm(span=5, adjust=False).mean()
    df['positive_momentum'] = ((df['ema_high'] - df['ema_close']) * df['volume']).apply(lambda x: max(x, 0))
    df['negative_momentum'] = ((df['low'] - df['ema_close']) * df['volume']).apply(lambda x: min(x, 0))
    df['smoothed_positive_momentum'] = df['positive_momentum'].ewm(alpha=smoothing_factor).mean()
    df['smoothed_negative_momentum'] = df['negative_momentum'].ewm(alpha=smoothing_factor).mean()
    df['momentum_indicator'] = df['smoothed_positive_momentum'] - df['smoothed_negative_momentum']
    
    # Final Alpha Factor
    df['smoothed_returns'] = df['close'].pct_change().ewm(alpha=smoothing_factor).mean()
    df['alpha_factor'] = df['smoothed_returns'] + df['complexity_score'] + df['momentum_indicator'] + 0.01
    
    return df['alpha_factor'].dropna()

# Example usage:
# df = pd.read_csv('stock_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df)
