import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Price Returns
    df['daily_return'] = df['close'] - df['close'].shift(1)
    
    # Identify Volume Spike Days
    df['volume_ema_20'] = df['volume'].ewm(span=20, adjust=False).mean()
    
    # Identify Amount Spike Days
    df['amount_ema_20'] = df['amount'].ewm(span=20, adjust=False).mean()
    
    # Combine Price Returns with Volume and Amount Spikes
    conditions = [
        (df['volume'] > 3 * df['volume_ema_20']) & (df['amount'] > 3 * df['amount_ema_20']),
        (df['volume'] > 2 * df['volume_ema_20']) | (df['amount'] > 2 * df['amount_ema_20'])
    ]
    choices = [df['daily_return'] * 5, df['daily_return'] * 4]
    df['combined_return'] = pd.np.select(conditions, choices, default=df['daily_return'])
    
    # Calculate Volume-Weighted Intraday High-Low Spread
    df['intraday_range'] = df['high'] - df['low']
    df['volume_weighted_intraday_range'] = df['intraday_range'] * df['volume']
    
    # Calculate Volume-Adjusted Opening Gap
    df['opening_gap'] = df['open'] - df['close'].shift(1)
    df['volume_adjusted_opening_gap'] = df['opening_gap'] * df['volume']
    
    # Combine Weighted Intraday High-Low Spread with Volume-Adjusted Opening Gap
    df['combined_value'] = df['volume_weighted_intraday_range'] + df['volume_adjusted_opening_gap']
    
    # Calculate Short-Term and Long-Term Exponential Moving Averages of Combined Value
    df['short_term_ema_12'] = df['combined_value'].ewm(span=12, adjust=False).mean()
    df['long_term_ema_26'] = df['combined_value'].ewm(span=26, adjust=False).mean()
    
    # Calculate Divergence
    df['divergence'] = df['short_term_ema_12'] - df['long_term_ema_26']
    
    # Apply Sign Function to Divergence
    df['divergence_sign'] = df['divergence'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
    # Calculate Daily Price Range
    df['daily_price_range'] = df['high'] - df['low']
    
    # Compute Volume-Adjusted Momentum
    df['momentum'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    df['volume_adjusted_momentum'] = df['momentum'] * df['volume']
    
    # Integrate High-Low Spread and Volume-Adjusted Momentum
    df['integrated_value'] = df['volume_adjusted_momentum'] * df['intraday_range']
    
    # Consider Directional Bias
    df['directional_bias'] = df.apply(lambda row: 1 if row['close'] > row['open'] else (-1 if row['close'] < row['open'] else 0), axis=1)
    
    # Combine All Components
    df['alpha_factor'] = (df['combined_return'] + df['integrated_value'] + df['divergence_sign']) * df['directional_bias']
    
    return df['alpha_factor']

# Example usage:
# alpha_factor_series = heuristics_v2(df)
