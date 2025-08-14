import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Compute Price Change
    df['price_change'] = df['close'].diff()
    
    # Detect Significant Volume Increase
    df['avg_volume_10'] = df['volume'].rolling(window=10).mean()
    df['volume_spike'] = df['volume'] > 2 * df['avg_volume_10']
    
    # Normalize Price Change by Intraday Volatility
    df['normalized_price_change'] = df['price_change'] / df['intraday_range']
    
    # Apply Volume-Weighted Adjustment
    df['adjusted_price_change'] = np.where(
        df['volume_spike'],
        df['normalized_price_change'] * 2,
        df['normalized_price_change']
    )
    
    # Accumulate Momentum Score over N days (e.g., 10 days)
    df['momentum_score'] = df['adjusted_price_change'].rolling(window=10).sum()
    
    # Calculate Rate of Change (ROC) over 14 days
    df['roc_14'] = (df['close'] - df['close'].shift(14)) / df['close'].shift(14)
    
    # Calculate Average True Range (ATR) over 10 days
    df['true_range'] = df[['high' - 'low', 'high' - df['close'].shift(1), 'low' - df['close'].shift(1)]].max(axis=1)
    df['atr_10'] = df['true_range'].rolling(window=10).mean()
    
    # Combine Momentum, Volatility, and Volume
    df['composite_alpha_factor'] = (df['momentum_score'] + df['roc_14'] + df['atr_10']) / 3
    
    # Apply Threshold (e.g., 75th percentile)
    threshold = df['composite_alpha_factor'].quantile(0.75)
    df['composite_alpha_factor'] = df['composite_alpha_factor'].where(df['composite_alpha_factor'] > threshold, other=np.nan)
    
    return df['composite_alpha_factor']
