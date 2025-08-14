import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the daily logarithmic returns for stability
    daily_log_returns = np.log(df['close']).diff(1)
    
    # Calculate the 5-day moving average of volume to identify trading activity anomalies
    avg_volume_5d = df['volume'].rolling(window=5).mean()
    volume_ratio = df['volume'] / avg_volume_5d
    
    # Calculate the true range (high - low)
    true_range = df['high'] - df['low']
    
    # Calculate the 5-day rolling standard deviation of the daily log returns to incorporate short-term volatility
    short_term_volatility = daily_log_returns.rolling(window=5).std()
    
    # Calculate the 20-day rolling standard deviation of the daily log returns to incorporate longer-term volatility
    long_term_volatility = daily_log_returns.rolling(window=20).std()
    
    # Calculate the 1-day momentum to capture short-term trends
    short_term_momentum = df['close'].diff(1)
    
    # Calculate the 10-day momentum to capture medium-term trends
    medium_term_momentum = df['close'].diff(10)
    
    # Calculate the 20-day momentum to capture longer-term trends
    long_term_momentum = df['close'].diff(20)
    
    # Calculate the adaptive weights based on the ratio of short-term to long-term momentum
    adaptive_weights = (short_term_momentum.abs() / (long_term_momentum.abs() + 1e-7)).clip(0, 1)
    
    # Combine daily log returns, short-term momentum, medium-term momentum, and long-term momentum, adaptively weighted
    combined_momentum = (daily_log_returns * adaptive_weights) + \
                        (short_term_momentum * (1 - adaptive_weights) * 0.3) + \
                        (medium_term_momentum * (1 - adaptive_weights) * 0.4) + \
                        (long_term_momentum * (1 - adaptive_weights) * 0.3)
    
    # Incorporate liquidity by using the average volume over the past 5 days
    liquidity_factor = 1 / avg_volume_5d
    
    # Final factor, adjusted by the true range, both short-term and long-term volatility, and liquidity
    factor = (combined_momentum * volume_ratio * liquidity_factor) / (true_range + 1e-7)
    factor = factor / (short_term_volatility + long_term_volatility + 1e-7)
    
    # Introduce a market microstructure adjustment using the close-to-open ratio
    close_to_open_ratio = df['close'] / df['open']
    factor = factor * close_to_open_ratio
    
    # Add sentiment analysis (assuming `sentiment_score` is a column in the DataFrame)
    sentiment_adjustment = 1 + (df['sentiment_score'] - df['sentiment_score'].mean()) / df['sentiment_score'].std()
    factor = factor * sentiment_adjustment
    
    # Add sector-specific factors (assuming `sector` is a column in the DataFrame)
    sector_mean_returns = df.groupby('sector')['close'].pct_change().groupby(df['sector']).transform('mean')
    sector_adjustment = 1 + (daily_log_returns - sector_mean_returns) / sector_mean_returns.abs()
    factor = factor * sector_adjustment
    
    return factor
