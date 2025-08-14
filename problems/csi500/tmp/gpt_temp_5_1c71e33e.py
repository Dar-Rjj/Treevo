import pandas as pd
import pandas as pd

def heuristics_v2(df, lookback_period):
    # Calculate Simple Moving Average of Close Prices
    sma = df['close'].rolling(window=lookback_period).mean()
    
    # Compute True Range
    df['prev_close'] = df['close'].shift(1)
    df['high_low'] = df['high'] - df['low']
    df['high_prev_close'] = (df['high'] - df['prev_close']).abs()
    df['low_prev_close'] = (df['low'] - df['prev_close']).abs()
    df['true_range'] = df[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
    
    # Apply Volume Weighting
    df['volume_weighted_true_range'] = df['volume'] * df['true_range']
    sum_volume_weighted_true_range = df['volume_weighted_true_range'].rolling(window=lookback_period).sum()
    sum_volume = df['volume'].rolling(window=lookback_period).sum()
    volume_adjusted_volatility = sum_volume_weighted_true_range / sum_volume
    
    # Compute Price Momentum
    price_momentum = df['close'] - sma
    
    # Final Alpha Factor
    alpha_factor = price_momentum / volume_adjusted_volatility
    
    return alpha_factor
