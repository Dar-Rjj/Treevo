import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Calculate 10-Day Sum of High-Low Ranges
    df['sum_high_low_range_10'] = df['high_low_range'].rolling(window=10).sum()
    
    # Calculate Price Change over 10 Days
    df['price_change_10'] = df['close'] - df['close'].shift(10)
    
    # Calculate Price Momentum
    df['price_momentum'] = df['close'] - df['close'].shift(1)
    
    # Classify Volume
    lookback_period = 30
    df['avg_volume'] = df['volume'].rolling(window=lookback_period).mean()
    df['volume_class'] = (df['volume'] > df['avg_volume']).astype(int)
    
    # Classify Amount
    df['avg_amount'] = df['amount'].rolling(window=lookback_period).mean()
    df['amount_class'] = (df['amount'] > df['avg_amount']).astype(int)
    
    # Assign Weights Based on Volume and Amount Classification
    conditions = [
        (df['volume_class'] == 1) & (df['amount_class'] == 1),
        (df['volume_class'] == 1) & (df['amount_class'] == 0),
        (df['volume_class'] == 0) & (df['amount_class'] == 1),
        (df['volume_class'] == 0) & (df['amount_class'] == 0)
    ]
    choices = [1.5, 1.0, 1.0, 0.5]
    df['weight'] = pd.np.select(conditions, choices, default=1.0)
    
    # Final Factor Calculation
    df['momentum_score'] = df['price_momentum'] * df['weight']
    
    # Volume Trend
    m_days = 21
    df['volume_trend'] = df['volume'].rolling(window=m_days).mean().pct_change()
    df['volume_trend_smoothed'] = df['volume_trend'].rolling(window=21).mean()
    df['volume_score'] = df['volume_trend_smoothed'].diff()
    
    # Integrate All Scores
    df['alpha_factor'] = (df['momentum_score'] * df['volume_score']) + df['sum_high_low_range_10']
    
    return df['alpha_factor'].dropna()

# Example usage:
# df = pd.read_csv('stock_data.csv', parse_dates=['date'], index_col='date')
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
