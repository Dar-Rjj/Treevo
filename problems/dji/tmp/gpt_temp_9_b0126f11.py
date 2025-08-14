import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Short-Term EMA (12-day span)
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    
    # Calculate Long-Term EMA (26-day span)
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # Calculate EMA Difference
    df['EMA_diff'] = df['EMA_12'] - df['EMA_26']
    
    # Calculate Volume-Weighted EMA Difference
    df['avg_volume_12'] = df['volume'].rolling(window=12).mean()
    df['volume_weighted_EMA_diff'] = df['EMA_diff'] * df['avg_volume_12']
    
    # Calculate Momentum Signal
    df['momentum_signal'] = df['volume_weighted_EMA_diff'] * df['volume']
    
    # Calculate Short-Term and Long-Term Momentum
    df['short_term_momentum'] = df['close'].shift(1) - df['close'].shift(7)
    df['long_term_momentum'] = df['close'].shift(1) - df['close'].shift(30)
    
    # Adjust Momentum by Volume Shock
    df['daily_volume_change'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    df['adjusted_momentum'] = (df['short_term_momentum'] - df['long_term_momentum']) * df['daily_volume_change']
    
    # Evaluate Weighted Average Volume Over Period
    df['weighted_avg_volume'] = df['volume'].rolling(window=12).sum() / 12
    
    # Calculate Volume-Adjusted Momentum
    df['vol_adj_momentum'] = (df['close'] - df['close'].shift(1)) / df['weighted_avg_volume']
    
    # Assess Positive vs Negative Momentum Contribution
    df['positive_mom_contribution'] = df.apply(lambda row: 1.5 * row['vol_adj_momentum'] if row['close'] > row['close'].shift(1) else row['vol_adj_momentum'], axis=1)
    
    # Integrate Accumulated Momentum Impact
    df['integrated_momentum'] = 0.4 * df['short_term_momentum'] + 0.6 * df['long_term_momentum']
    
    # Synthesize Final Alpha Factor
    df['alpha_factor'] = 0.7 * df['momentum_signal'] + 0.3 * df['integrated_momentum']
    
    return df['alpha_factor']
