import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Return
    df['intraday_return'] = df['close'] - df['open']
    
    # Calculate Intraday High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Combine Intraday Return and High-Low Range
    df['combined_factor'] = df['intraday_return'] * df['high_low_range']
    df['smoothed_factor'] = df['combined_factor'].ewm(span=14, adjust=False).mean()
    
    # Apply Volume Weighting
    df['volume_weighted_factor'] = df['smoothed_factor'] * df['volume']
    
    # Incorporate Previous Day's Closing Gap
    df['prev_close_gap'] = df['open'] - df['close'].shift(1)
    df['volume_weighted_with_gap'] = df['volume_weighted_factor'] + df['prev_close_gap']
    
    # Integrate Long-Term Momentum
    df['long_term_return'] = df['close'] - df['close'].shift(50)
    df['normalized_long_term_return'] = df['long_term_return'] / df['high_low_range']
    
    # Include Enhanced Dynamic Volatility Component
    df['intraday_returns_std'] = df['intraday_return'].rolling(window=20).std()
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['combined_volatility'] = (df['intraday_returns_std'] + df['atr']) / 2
    df['volatility_adjusted'] = df['combined_volatility'] * df['volume']
    
    # Integrate Sentiment Analysis
    # Assuming sentiment scores are available in a column named 'sentiment_score'
    sentiment_constant = 0.5
    df['sentiment_adjusted'] = df['sentiment_score'] * sentiment_constant
    
    # Incorporate Sector Performance
    # Assuming we have a separate DataFrame `sector_df` with sector data
    sector_return = (sector_df['average_close_price'] - sector_df['average_close_price'].shift(1)) / sector_df['high_low_range']
    df['normalized_sector_return'] = sector_return.reindex(df.index, method='ffill')
    
    # Final Factor Calculation
    df['final_factor'] = (
        df['volume_weighted_with_gap'] +
        df['normalized_long_term_return'] +
        df['volatility_adjusted'] +
        df['sentiment_adjusted'] +
        df['normalized_sector_return']
    )
    
    # Apply Non-Linear Transformation
    df['final_factor_transformed'] = np.log1p(df['final_factor'])
    
    return df['final_factor_transformed']

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# final_factor = heuristics_v2(df)
