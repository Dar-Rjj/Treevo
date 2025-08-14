import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Trend Analysis
    df['price_movement'] = df['close'].diff()
    df['daily_volatility'] = df['high'] - df['low']
    
    # Average True Range (ATR) without normalization
    df['tr'] = df[['high', 'low']].apply(lambda x: max(x['high'], df['close'].shift(1)) - min(x['low'], df['close'].shift(1)), axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    # Volume Impact on Prices
    df['volume_change'] = df['volume'].diff()
    df['price_volume_ratio'] = df['price_movement'] / df['volume_change'].replace(0, 1)  # Avoid division by zero
    
    # Momentum Indicators
    short_term_window = 5
    long_term_window = 20
    df['short_term_sum'] = df['close'].rolling(window=short_term_window).sum()
    df['long_term_sum'] = df['close'].rolling(window=long_term_window).sum()
    df['momentum'] = df['short_term_sum'] - df['long_term_sum']
    
    # Support and Resistance Levels
    df['recent_high'] = df['high'].rolling(window=20).max()
    df['recent_low'] = df['low'].rolling(window=20).min()
    df['test_high_count'] = df['high'].rolling(window=20).apply(lambda x: (x == x.max()).sum(), raw=True)
    df['test_low_count'] = df['low'].rolling(window=20).apply(lambda x: (x == x.min()).sum(), raw=True)
    
    # Breakout Potential
    n_days = 20
    df['distance_to_highest_high'] = df['high'].rolling(window=n_days).max() - df['close']
    df['distance_to_lowest_low'] = df['close'] - df['low'].rolling(window=n_days).min()
    
    # Accumulation and Distribution
    df['money_flow_multiplier'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['money_flow_volume'] = df['money_flow_multiplier'] * df['volume']
    df['accumulation_distribution'] = df['money_flow_volume'].rolling(window=14).sum()
    
    # Volatility and Risk
    df['volatility'] = df['close'].rolling(window=20).std()
    df['closing_range'] = df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()
    
    # Seasonal and Cyclical Patterns
    df['month'] = df.index.month
    monthly_returns = df['close'].resample('M').last().pct_change()
    df = df.merge(monthly_returns, left_index=True, right_index=True, how='left')
    df.rename(columns={'close_y': 'monthly_return'}, inplace=True)
    
    # Relative Strength
    # Assuming 'benchmark' is a column in the DataFrame representing the benchmark index
    df['relative_strength'] = df['close_x'].pct_change() / df['benchmark'].pct_change()
    
    # Price and Volume Ratios
    df['pv_ratio'] = df['close'].rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # Final alpha factor
    alpha_factor = (df['price_movement'] + 
                    df['momentum'] + 
                    df['accumulation_distribution'] + 
                    df['relative_strength'] + 
                    df['pv_ratio']).fillna(0)
    
    return alpha_factor
