import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Momentum-based Alpha Factor
    df['daily_return'] = df['close'].pct_change()
    df['weekly_return'] = df['close'].pct_change(periods=5)  # Assuming 5 business days in a week
    
    # EMA based momentum
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['ema_momentum'] = df['ema12'] - df['ema26']
    
    # Aggregate Momentum Score
    df['momentum_score'] = df['daily_return'] + df['weekly_return'] + df['ema_momentum']
    
    # Volume-based Alpha Factor
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['avg_volume_30d'] = df['volume'].rolling(window=30).mean()
    df['volume_change_indicator'] = df['volume'] - df['avg_volume_30d']
    
    # Combined Volume Factor
    df['combined_volume_factor'] = df['obv'] + df['volume_change_indicator']
    
    # Reversal-based Alpha Factor
    df['return_5d'] = df['close'].pct_change().rolling(window=5).std()
    df['return_20d'] = df['close'].pct_change().rolling(window=20).std()
    df['short_term_mean_reversion'] = df['return_5d'].apply(lambda x: 1 if x < df['daily_return'] - 1 else 0)
    df['long_term_mean_reversion'] = df['return_20d'].apply(lambda x: -1 if x > df['daily_return'] + 1 else 0)
    
    # Combine Reversal Signals
    df['reversal_score'] = df['short_term_mean_reversion'] + df['long_term_mean_reversion']
    
    # Integrate Different Alpha Factors
    # Simple Linear Combination with weights
    weights = {'momentum': 0.4, 'volume': 0.3, 'reversal': 0.3}  # Example weights
    df['composite_alpha'] = (weights['momentum'] * df['momentum_score'] +
                            weights['volume'] * df['combined_volume_factor'] +
                            weights['reversal'] * df['reversal_score'])
    
    return df['composite_alpha']

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# alpha_values = heuristics_v2(df)
# print(alpha_values)
