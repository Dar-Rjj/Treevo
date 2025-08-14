import pandas as pd
import pandas as pd

def heuristics_v2(df, N=5):
    # Calculate Price-Volume Momentum
    close_sma = df['close'].rolling(window=N).mean()
    price_difference = df['close'] - close_sma
    momentum_score = price_difference / close_sma
    volume_sum = df['volume'].rolling(window=N).sum()
    adjusted_momentum_score = momentum_score * volume_sum
    
    # Calculate Volume-Weighted Factor
    daily_returns = (df['high'] - df['close'].shift(1)) * df['volume']
    aggregate_product = daily_returns.rolling(window=N).sum()
    final_volume_weighted_factor = aggregate_product / volume_sum
    
    # Adjust Momentum Score by Volume-Weighted Factor
    adjusted_momentum_score *= final_volume_weighted_factor
    
    # Calculate High-to-Low Price Range
    high_low_range = df['high'] - df['low']
    
    # Calculate Open-Adjusted Range
    high_open_diff = df['high'] - df['open']
    open_low_diff = df['open'] - df['low']
    open_adjusted_range = pd.Series([max(ho, ol) for ho, ol in zip(high_open_diff, open_low_diff)])
    
    # Calculate High-to-Low Range Relative to Open
    high_open_rel = (df['high'] - df['open']) / df['open']
    open_low_rel = (df['open'] - df['low']) / df['open']
    high_low_rel_to_open = pd.Series([max(ho, ol) for ho, ol in zip(high_open_rel, open_low_rel)])
    
    # Calculate Trading Intensity
    volume_change = df['volume'] - df['volume'].shift(1)
    amount_change = df['amount'] - df['amount'].shift(1)
    trading_intensity = volume_change / amount_change
    
    # Combine All Factors
    factor = (
        adjusted_momentum_score +
        high_low_range +
        open_adjusted_range +
        high_low_rel_to_open
    )
    
    return factor
