import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    close_to_open_return = df['close'].pct_change()
    
    # Adaptive Weighting
    volume_weight = df['volume'] / df['volume'].rolling(window=20).mean()
    recent_data_importance = 1 / (df.index.to_series().diff().dt.days + 1)
    
    adaptive_weight = 0.5 * volume_weight + 0.5 * recent_data_importance
    
    # Dynamic Volatility Adjustment
    volatility_adjustment = (intraday_range + abs(close_to_open_return)) / 2
    
    # Combine Intraday Range and Momentum
    momentum = close_to_open_return.rolling(window=20).mean()
    factor_value = (intraday_range * adaptive_weight * volatility_adjustment) + (momentum * (1 - adaptive_weight) * volatility_adjustment)
    
    return pd.Series(factor_value, index=df.index, name='alpha_factor')

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
