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
    avg_volume = df['volume'].rolling(window=20).mean()
    df['significant_volume_increase'] = (df['volume'] > 2 * avg_volume).astype(int)
    
    # Normalize Price Change by Intraday Volatility
    df['normalized_price_change'] = df['price_change'] / df['intraday_range']
    
    # Apply Volume-Weighted Adjustment
    df['adjusted_price_change'] = np.where(
        df['significant_volume_increase'] == 1,
        2 * df['normalized_price_change'],
        df['normalized_price_change']
    )
    
    # Accumulate Momentum Score over N days
    df['momentum_score'] = df['adjusted_price_change'].rolling(window=20).sum()
    
    # Refine Momentum Component
    df['short_term_roc'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    df['long_term_roc'] = (df['close'] - df['close'].shift(28)) / df['close'].shift(28)
    df['combined_roc'] = (df['short_term_roc'] + df['long_term_roc']) / 2
    
    # Incorporate Average True Range (ATR)
    df['true_range'] = df[['high' - 'low', 
                           abs(df['high'] - df['close'].shift(1)), 
                           abs(df['low'] - df['close'].shift(1))]].max(axis=1)
    df['atr_10'] = df['true_range'].rolling(window=10).mean()
    
    # Integrate Trading Volume as a Measure of Interest
    df['volume_moving_avg_20'] = df['volume'].rolling(window=20).mean()
    df['volume_percentage_change'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Introduce Open-Close Price Relationship Factor
    df['daily_return'] = (df['close'] - df['open']) / df['open']
    df['std_daily_return_20'] = df['daily_return'].rolling(window=20).std()
    
    # Combine Factors into a Composite Alpha Factor
    df['composite_alpha_factor'] = (
        df['combined_roc'] + 
        df['atr_10'] + 
        df['volume_percentage_change'] + 
        df['std_daily_return_20']
    ) / 4
    
    # Apply Threshold to Filter Out Signals
    threshold = df['composite_alpha_factor'].quantile(0.75)  # Example: 75th percentile
    df['final_alpha_factor'] = np.where(
        df['composite_alpha_factor'] > threshold,
        df['composite_alpha_factor'],
        0
    )
    
    return df['final_alpha_factor']
