import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Compute Price Change
    df['price_change'] = df['close'] - df['close'].shift(1)
    
    # Detect Significant Volume Increase
    df['avg_volume_20'] = df['volume'].rolling(window=20).mean()
    df['volume_spike'] = df['volume'] > 2 * df['avg_volume_20']
    
    # Normalize Price Change by Intraday Volatility
    df['normalized_price_change'] = df['price_change'] / df['intraday_range']
    
    # Apply Volume-Weighted Adjustment
    df['adjusted_price_change'] = df['normalized_price_change'] * (df['volume_spike'] + 1)
    
    # Accumulate Momentum Score over N days (N=10 for example)
    df['momentum_score'] = df['adjusted_price_change'].rolling(window=10).sum()
    
    # Calculate Short-Term Rate of Change (ROC) over 5 days
    df['roc_short_term'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Calculate Long-Term Rate of Change (ROC) over 28 days
    df['roc_long_term'] = (df['close'] - df['close'].shift(28)) / df['close'].shift(28)
    
    # Combine Short-Term and Long-Term ROCs
    df['momentum_factor'] = (df['roc_short_term'] + df['roc_long_term']) / 2
    
    # Calculate Average True Range (ATR) over 10 days
    df['true_range'] = df[['high', 'low']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - df['close'].shift(1)), abs(x['low'] - df['close'].shift(1))), axis=1)
    df['atr_10'] = df['true_range'].rolling(window=10).mean()
    
    # Integrate Cumulative Enhanced Momentum over 28 days
    df['weighted_adjusted_price_change'] = df['volume'] * df['adjusted_price_change']
    df['cumulative_enhanced_momentum'] = df['weighted_adjusted_price_change'].rolling(window=28).sum()
    
    # Calculate Relative Strength over 28 days
    df['min_close_28'] = df['close'].rolling(window=28).min()
    df['max_close_28'] = df['close'].rolling(window=28).max()
    df['relative_strength'] = (df['close'] - df['min_close_28']) / (df['max_close_28'] - df['min_close_28'])
    
    # Calculate Daily Return using Open and Close Prices
    df['daily_return'] = (df['close'] - df['open']) / df['open']
    df['std_daily_return_22'] = df['daily_return'].rolling(window=22).std()
    
    # Combine Factors into a Composite Alpha Factor
    df['composite_alpha_factor'] = (df['momentum_factor'] + df['cumulative_enhanced_momentum'] + df['relative_strength'] + df['atr_10']) / 4
    
    # Apply Threshold: Only consider stocks where the composite alpha factor exceeds a certain percentile of all stocks' values
    threshold_percentile = 75  # Example: 75th percentile
    df['alpha_factor'] = df['composite_alpha_factor'] > df['composite_alpha_factor'].quantile(threshold_percentile / 100)
    
    return df['alpha_factor']
