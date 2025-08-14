import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Volume-Weighted Price Returns
    df['open_t1'] = df['open'].shift(-1)
    df['simple_returns'] = (df['open_t1'] - df['close']) / df['close']
    df['volume_weighted_returns'] = df['simple_returns'] * df['volume']
    
    # Identify Volume Surge Days
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    df['rolling_avg_volume'] = df['volume'].rolling(window=5).mean()
    df['is_surge_day'] = (df['volume'] > df['rolling_avg_volume']).astype(bool)
    
    # Calculate Adaptive Volatility
    df['daily_returns'] = df['close'].pct_change()
    lookback_period = 20 + (60 - 20) * (df['daily_returns'].std() / df['daily_returns'].std().rolling(window=60).mean())
    df['volatility'] = df['daily_returns'].rolling(window=lookback_period.astype(int)).std()
    df['volume_moving_avg'] = df['volume'].rolling(window=20).mean()
    df['volume_std'] = df['volume'].rolling(window=20).std()
    df['volume_z_score'] = (df['volume'] - df['volume_moving_avg']) / df['volume_std']
    df['adjusted_volatility'] = df['volatility'] * (1 + np.abs(df['volume_z_score']))
    
    # Refine Surge Factors
    df['volume_surge_ratio'] = df['volume'] / df['volume'].shift(1)
    conditions = [
        (df['volume_surge_ratio'] > 2.5),
        (df['volume_surge_ratio'] > 2.0) & (df['volume_surge_ratio'] <= 2.5),
        (df['volume_surge_ratio'] > 1.5) & (df['volume_surge_ratio'] <= 2.0)
    ]
    choices = [1.8, 1.5, 1.2]
    df['refined_surge_factor'] = np.select(conditions, choices, default=1.0)
    
    # Adjust Volume-Weighted Returns by Adaptive Volatility
    df['adjusted_returns'] = df['volume_weighted_returns'] / df['adjusted_volatility']
    
    # Combine Adjusted Returns with Refined Volume Surge Indicator
    df['final_factor'] = df['adjusted_returns'] * np.where(df['is_surge_day'], df['refined_surge_factor'], 1.0)
    
    return df['final_factor'].dropna()

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# factor_values = heuristics_v2(df)
