import pandas as pd
import pandas as pd

def heuristics_v2(df, n_days=10, threshold_percentile=75):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Compute Price Change
    df['price_change'] = df['close'].diff()
    
    # Detect Significant Volume Increase
    avg_volume = df['volume'].rolling(window=n_days).mean()
    df['significant_volume_increase'] = (df['volume'] > 2 * avg_volume)
    
    # Normalize Price Change by Intraday Volatility
    df['normalized_price_change'] = df['price_change'] / df['intraday_range']
    
    # Apply Volume-Weighted Adjustment
    df['adjusted_price_change'] = df.apply(
        lambda row: row['normalized_price_change'] * 2 if row['significant_volume_increase'] else row['normalized_price_change'],
        axis=1
    )
    
    # Accumulate Momentum Score
    df['momentum_score'] = df['adjusted_price_change'].rolling(window=n_days).sum()
    
    # Calculate Rate of Change (ROC)
    df['roc'] = (df['close'] - df['close'].shift(14)) / df['close'].shift(14)
    
    # Calculate True Range
    df['true_range'] = df.apply(
        lambda row: max(row['high'] - row['low'], abs(row['high'] - row['close'].shift(1)), abs(row['low'] - row['close'].shift(1))),
        axis=1
    )
    
    # Calculate Average True Range (ATR)
    df['atr'] = df['true_range'].rolling(window=10).mean()
    
    # Combine Momentum, Volatility, and Volume
    df['composite_alpha_factor'] = df['momentum_score'] + df['roc'] + df['atr']
    df['composite_alpha_factor'] = df.apply(
        lambda row: row['composite_alpha_factor'] * 1.5 if row['significant_volume_increase'] else row['composite_alpha_factor'],
        axis=1
    )
    
    # Apply Threshold
    percentile_value = df['composite_alpha_factor'].quantile(threshold_percentile / 100)
    df['composite_alpha_factor'] = df['composite_alpha_factor'].where(df['composite_alpha_factor'] > percentile_value, other=0)
    
    return df['composite_alpha_factor']
