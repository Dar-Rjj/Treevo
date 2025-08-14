import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate 5-Day Price Momentum
    df['5_day_price_momentum'] = df['close'].pct_change(5)
    
    # Calculate Volume Change
    df['volume_change'] = df['volume'].pct_change()
    
    # Calculate Amount Change
    df['amount_change'] = df['amount'].pct_change()
    
    # Define Positive Volume Change Threshold and Positive Amount Change Threshold
    volume_threshold = 0.01
    amount_threshold = 0.01
    
    # Apply Thresholds to Volume and Amount Changes
    df['price_momentum_factor'] = df.apply(
        lambda row: row['5_day_price_momentu'] if (row['volume_change'] > volume_threshold) and (row['amount_change'] > amount_threshold) else 0, axis=1
    )
    
    # Calculate Trend Momentum Indicator
    df['20_day_ema'] = df['close'].ewm(span=20).mean()
    df['20_day_std'] = df['close'].rolling(window=20).std()
    df['trend_momentum_indicator'] = (df['close'] - df['20_day_ema']) / df['20_day_std']
    
    # Calculate Intraday High-Low Spread
    df['intraday_high_low_spread'] = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = (df['close'] - df['open']) / df['open']
    
    # Calculate Volume Weighted Volatility
    df['daily_log_return'] = df['close'].pct_change().apply(lambda x: np.log(1 + x))
    df['absolute_log_return'] = df['daily_log_return'].abs()
    df['volume_weighted_volatility'] = (df['absolute_log_return'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # Combine Intraday Momentum with Volume Weighted Volatility
    df['intraday_momentum_volatility'] = (df['intraday_high_low_spread'] * df['close_to_open_return']) / df['volume_weighted_volatility']
    
    # Calculate High-Low Momentum
    N = 20
    df['high_low_momentum'] = (df['high'] - df['low']).rolling(window=N).sum() / N
    
    # Calculate Volume Change
    M = 20
    df['avg_volume'] = df['volume'].rolling(window=M).mean()
    df['volume_change_factor'] = df.apply(
        lambda row: 1 if row['volume'] > row['avg_volume'] else -1, axis=1
    )
    
    # Combine High-Low Momentum and Volume Change
    df['combined_momentum_factor'] = df['high_low_momentum'] * df['volume_change_factor']
    
    # Final Crossover Alpha Factor
    df['crossover_alpha_factor'] = df['trend_momentum_indicator'] + df['intraday_momentum_volatility'] + df['combined_momentum_factor']
    
    return df['crossover_alpha_factor']
