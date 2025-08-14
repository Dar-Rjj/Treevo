import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily returns
    df['daily_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Compute rolling standard deviation of daily returns with a 20-day window
    df['volatility'] = df['daily_return'].rolling(window=20).std()
    
    # Create a smoothed volatility using exponential moving average with a 10-day EMA
    df['smoothed_volatility'] = df['volatility'].ewm(span=10, adjust=False).mean()
    
    # Adaptive Volatility Factor
    adaptive_volatility_factor = df['smoothed_volatility']
    
    # Calculate short-term momentum
    df['short_term_momentum'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    
    # Calculate medium-term momentum
    df['medium_term_momentum'] = (df['close'] - df['close'].shift(60)) / df['close'].shift(60)
    
    # Calculate long-term momentum
    df['long_term_momentum'] = (df['close'] - df['close'].shift(120)) / df['close'].shift(120)
    
    # Multi-Period Momentum Factor (using equal weights for simplicity)
    df['multi_period_momentum'] = (df['short_term_momentum'] + df['medium_term_momentum'] + df['long_term_momentum']) / 3
    
    # Calculate volume-weighted price
    df['volume_weighted_price'] = (df['high'] + df['low'] + df['close']) / 3 * df['volume']
    
    # Compute the cumulative sum of volume-weighted price with a 20-day rolling window
    df['cumulative_volume_weighted_price'] = df['volume_weighted_price'].rolling(window=20).sum()
    
    # Create a trend score
    df['trend_score'] = (df['cumulative_volume_weighted_price'] - df['cumulative_volume_weighted_price'].shift(20)) / df['cumulative_volume_weighted_price'].shift(20)
    
    # Volume-Weighted Trend Factor
    volume_weighted_trend_factor = df['trend_score']
    
    # Sector Rotation Factor
    # Assuming 'sector' is a column in the dataframe and we have a function to calculate sector performance
    def calculate_sector_performance(group):
        return (group['close'] - group['close'].shift(1)).mean() / group['close'].shift(1).mean()
    
    sector_performance = df.groupby('sector').apply(calculate_sector_performance).reset_index(name='sector_performance')
    df = pd.merge(df, sector_performance, on='sector')
    
    # Market return
    market_return = (df['close'] - df['close'].shift(1)).mean() / df['close'].shift(1).mean()
    
    # Compare sector performance to the market
    df['relative_sector_performance'] = (df['sector_performance'] - market_return) / market_return
    
    # Sector Rotation Score
    df['sector_rotation_score'] = df['relative_sector_performance'].rolling(window=20).mean()
    sector_rotation_factor = df['sector_rotation_score']
    
    # Macro-Economic Indicator Factor
    # Assuming we have a DataFrame `macro_data` with columns: 'interest_rate', 'unemployment_rate', 'gdp_growth'
    # and it is merged with the main dataframe
    # Simple linear model for demonstration
    X = macro_data[['interest_rate', 'unemployment_rate', 'gdp_growth']]
    y = df['close'].pct_change().dropna()
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(X, y)
    macro_factors = model.predict(X)
    df['macro_economic_factor'] = macro_factors
    
    # Combine Factors
    # Using a simple equal-weighted sum for demonstration; in practice, use a machine learning model
    combined_factor = (adaptive_volatility_factor + df['multi_period_momentum'] + volume_weighted_trend_factor + sector_rotation_factor + df['macro_economic_factor']) / 5
    
    return combined_factor
