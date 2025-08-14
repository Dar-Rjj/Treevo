import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Momentum
    df['intraday_momentum'] = (df['close'] - df['open']) / df['open']
    
    # Identify Volume Acceleration
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    df['volume_change_rate'] = df['volume_change'] / df['volume'].shift(1)
    
    # Assign Trend Strength
    trend_strength = df['volume_change_rate'].apply(
        lambda x: 1.5 if x > 0.1 else (0.5 if x < -0.1 else 1.0)
    )
    
    # Integrate Intraday Momentum and Volume Trend
    df['integrated_factor'] = df['intraday_momentum'] * trend_strength
    
    # Consider Price Volatility
    df['true_range'] = df.apply(
        lambda row: max(
            row['high'] - row['low'],
            abs(row['high'] - row['close'].shift(1)),
            abs(row['close'].shift(1) - row['low'])
        ),
        axis=1
    )
    
    # Adjust Alpha Factor based on True Range
    df['alpha_factor'] = df['integrated_factor'] / (1 + df['true_range'])
    
    return df['alpha_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date')
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
