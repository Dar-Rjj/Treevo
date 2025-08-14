import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate High-Low Spread
    high_low_spread = df['high'] - df['low']
    
    # Weight by Volume
    volume_weighted_spread = high_low_spread * df['volume']
    
    # Condition on Close-to-Open Return
    close_open_return = (df['close'] - df['open']) / df['open']
    positive_return_weight = 1.5
    negative_return_weight = 0.5
    weighted_spread = volume_weighted_spread * (positive_return_weight if close_open_return > 0 else negative_return_weight)
    
    # Introduce Amount Influence
    avg_amount_5d = df['amount'].rolling(window=5).mean()
    amount_influence = df['amount'].apply(lambda x: 1.1 if x > avg_amount_5d else 0.9)
    adjusted_weighted_spread = weighted_spread * amount_influence
    
    # Simple Price Momentum
    daily_returns = (df['close'] / df['close'].shift(1)) - 1
    simple_momentum = daily_returns.rolling(window=5).mean()
    
    # Volume-Weighted Momentum
    volume_weighted_returns = (daily_returns * df['volume']).rolling(window=5).sum() / df['volume'].rolling(window=5).sum()
    
    # High-Low Range Indicator
    high_low_diff = df['high'] - df['low']
    high_low_range_10d = high_low_diff.rolling(window=10).mean()
    
    # Volume-Weighted Open-Close Spread
    open_close_diff = df['close'] - df['open']
    volume_weighted_open_close = (open_close_diff * df['volume']).rolling(window=10).sum() / df['volume'].rolling(window=10).sum()
    
    # Synthesize with High-Low Spread Factors
    combined_momentum = 0.5 * (simple_momentum + volume_weighted_returns)
    
    # Adjust with Amount Influence
    momentum_adjusted = combined_momentum * amount_influence
    
    # Cumulative Return
    N = 10
    cumulative_return = (df['close'] / df['close'].shift(N)) - 1
    
    # Volume Influence
    avg_volume_Nd = df['volume'].rolling(window=N).mean()
    volume_influenced_return = cumulative_return * avg_volume_Nd
    
    # Price Range Adjustment
    max_high_Nd = df['high'].rolling(window=N).max()
    min_low_Nd = df['low'].rolling(window=N).min()
    price_range = max_high_Nd - min_low_Nd
    range_adjusted_return = volume_influenced_return / price_range
    
    # Combined Alpha Factor
    alpha_factor = 0.5 * (momentum_adjusted + range_adjusted_return)
    
    return alpha_factor.dropna()

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df)
