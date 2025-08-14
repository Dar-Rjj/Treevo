import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def adaptive_ema(series, short_window=5, long_window=20):
    # Simple example of determining which EMA to use based on recent volatility
    volatility = series.rolling(window=10).std()
    if volatility.iloc[-1] > volatility.median():
        return series.ewm(span=short_window).mean()
    else:
        return series.ewm(span=long_window).mean()

def volume_weighted_close(df):
    return (df['close'] * df['volume']) / df['volume']

def heuristics_v2(df, news_df, sector_returns, event_impact_scores, alpha=0.3):
    """
    Generate a novel and interpretable alpha factor that combines sentiment, sector performance, and news impact.
    
    Parameters:
    - df: DataFrame with market features (open, high, low, close, amount, volume)
    - news_df: DataFrame with daily sentiment scores indexed by date
    - sector_returns: DataFrame with sector returns indexed by date
    - event_impact_scores: DataFrame with event impact scores indexed by date
    - alpha: Smoothing factor for exponential smoothing
    
    Returns:
    - Series: Alpha factor values indexed by date
    """
    
    # Volume-Weighted Close Price
    df['vwc'] = volume_weighted_close(df)
    
    # Sentiment Analysis
    sentiment_ema_short = adaptive_ema(news_df['sentiment_score'], short_window=5, long_window=20)
    sentiment_ema_long = adaptive_ema(news_df['sentiment_score'], short_window=5, long_window=20)
    sentiment_factor = sentiment_ema_short - sentiment_ema_long
    
    # Sector Performance
    sector_ema_short = adaptive_ema(sector_returns, short_window=5, long_window=20)
    sector_ema_long = adaptive_ema(sector_returns, short_window=5, long_window=20)
    sector_factor = sector_ema_short - sector_ema_long
    
    # News Impact
    event_ema_short = adaptive_ema(event_impact_scores, short_window=5, long_window=20)
    event_ema_long = adaptive_ema(event_impact_scores, short_window=5, long_window=20)
    news_factor = event_ema_short - event_ema_long
    
    # Short-Term Factors
    high_low_range_daily = df['high'] - df['low']
    trading_volume_daily = df['volume']
    momentum_daily = df['close'] - df['open']
    
    # Long-Term Factors
    weekly_df = df.resample('W').agg({'high': 'max', 'low': 'min', 'open': 'first', 'close': 'last', 'volume': 'sum'})
    high_low_range_weekly = weekly_df['high'] - weekly_df['low']
    trading_volume_weekly = weekly_df['volume']
    momentum_weekly = weekly_df['close'] - weekly_df['open']
    
    # Combine all factors
    combined_factors = (
        sentiment_factor + 
        sector_factor + 
        news_factor + 
        high_low_range_daily + 
        trading_volume_daily + 
        momentum_daily + 
        high_low_range_weekly + 
        trading_volume_weekly + 
        momentum_weekly
    )
    
    # Exponential Smoothing
    smoothed_alpha = combined_factors.ewm(alpha=alpha).mean()
    
    # Z-Score normalization
    normalized_alpha = zscore(smoothed_alpha)
    
    return pd.Series(normalized_alpha, index=df.index)

# Example usage
# df, news_df, sector_returns, event_impact_scores should be pre-processed DataFrames
# alpha = 0.3  # You can adjust the smoothing factor
# result = heuristics_v2(df, news_df, sector_returns, event_impact_scores, alpha)
