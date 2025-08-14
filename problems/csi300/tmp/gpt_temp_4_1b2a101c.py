import numpy as np
def heuristics_v2(df):
    # Calculate High-Low Range Momentum
    df['high_low_range'] = df['high'] - df['low']
    df['range_momentum'] = (df['high_low_range'] > df['high_low_range'].shift(1)).astype(int)
    
    # Calculate Close-to-Low Spread
    df['close_to_low_spread'] = np.maximum(df['close'] - df['low'], 0)
    
    # Calculate Volume-Weighted High-Low Range
    df['volume_weighted_high_low'] = df['high_low_range'] * df['volume']
    
    # Calculate Volume-Adjusted Spread
    lookback_period = 10
    df['accumulated_spread'] = df['close_to_low_spread'].rolling(window=lookback_period).sum()
    df['total_volume'] = df['volume'].rolling(window=lookback_period).sum()
    df['volume_adjusted_spread'] = df['accumulated_spread'] / df['total_volume']
    
    # Combine High-Low Range Momentum and Volume-Adjusted Spread
    df['combined_momentum_spread'] = df['range_momentum'] * df['volume_adjusted_spread']
    
    # Calculate Momentum
    df['price_change'] = df['close'] - df['close'].shift(1)
    
    # Calculate Final Relative Strength Factor
    positive_changes = df['price_change'].apply(lambda x: x if x > 0 else 0)
    negative_changes = df['price_change'].apply(lambda x: x if x < 0 else 0)
    positive_sum = positive_changes.rolling(window=lookback_period).sum()
    negative_sum = negative_changes.abs().rolling(window=lookback_period).sum()
    df['relative_strength_ratio'] = (positive_sum / (negative_sum + 1e-6)).fillna(0)
    
    # Integrate Intraday Factors
    df['intraday_high_low_ratio'] = (df['high'] - df['low']) / df['low']
    df['close_to_open_return'] = (df['close'] - df['open']) / df['open']
