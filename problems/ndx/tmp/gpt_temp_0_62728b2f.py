import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Compute Price Change
    df['price_change'] = df['close'] - df['close'].shift(1)
    
    # Detect Significant Volume Increase
    df['avg_volume_5'] = df['volume'].rolling(window=5).mean()
    df['volume_spike'] = (df['volume'] > 2 * df['avg_volume_5']).astype(int)
    
    # Normalize Price Change by Intraday Volatility
    df['normalized_price_change'] = df['price_change'] / df['intraday_range']
    
    # Apply Volume-Weighted Adjustment
    df['adjusted_price_change'] = df['normalized_price_change'] * (1 + df['volume_spike'])
    
    # Accumulate Momentum Score
    df['momentum_score'] = df['adjusted_price_change'].rolling(window=5).sum()
    
    # Refine Momentum Component
    df['roc_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    df['roc_28'] = (df['close'] - df['close'].shift(28)) / df['close'].shift(28)
    df['combined_roc'] = (df['roc_5'] + df['roc_28']) / 2
    
    # Incorporate Average True Range (ATR)
    df['true_range'] = df[['high', 'low']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - df['close'].shift(1)), abs(x['low'] - df['close'].shift(1))), axis=1)
    df['atr_14'] = df['true_range'].rolling(window=14).mean()
    
    # Integrate Trading Volume as a Measure of Interest
    df['avg_volume_20'] = df['volume'].rolling(window=20).mean()
    df['volume_interest'] = (df['volume'] > df['avg_volume_20']).astype(int)
    
    # Introduce Open-Close Price Relationship Factor
    df['daily_return'] = (df['close'] - df['open']) / df['open']
    df['std_daily_return_30'] = df['daily_return'].rolling(window=30).std()
    
    # Combine Factors into a Composite Alpha Factor
    df['composite_alpha_factor'] = (
        0.3 * df['combined_roc'] + 
        0.2 * df['atr_14'] + 
        0.2 * df['volume_interest'] + 
        0.3 * df['std_daily_return_30']
    )
    
    # Apply Threshold to Filter Out Signals
    df['composite_alpha_factor'] = df['composite_alpha_factor'].where(df['composite_alpha_factor'] > df['composite_alpha_factor'].quantile(0.75), 0)
    
    return df['composite_alpha_factor']
