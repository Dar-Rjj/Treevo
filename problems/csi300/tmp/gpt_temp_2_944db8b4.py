import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Return
    df['daily_return'] = df['close'].pct_change()
    
    # Compute Exponential Moving Average (EMA) of Returns
    ema_span = 10
    df['ema_returns'] = df['daily_return'].ewm(span=ema_span, adjust=False).mean()
    
    # Identify High and Low Volatility Days
    # Calculate True Range
    df['true_range'] = df[['high', 'low', 'close']].apply(lambda x: max(x[0], x[1]) - min(x[0], x[1]), axis=1)
    df['true_range'] = df['true_range'].shift(1)  # Using previous day's close price for true range calculation
    
    # Calculate Average True Range (ATR)
    atr_period = 14
    df['atr'] = df['true_range'].rolling(window=atr_period).mean()
    
    # Categorize Days by Volatility
    volatility_threshold = df['atr'].mean()  # Example threshold, can be adjusted
    df['is_high_volatility'] = df['atr'] > volatility_threshold
    
    # Filter Days by Volatility
    high_vol_days = df[df['is_high_volatility']]
    low_vol_days = df[~df['is_high_volatility']]
    
    # Calculate Volume Weighted Momentum
    df['volume_weighted_momentum'] = df['daily_return'] * df['volume']
    
    # Compute Momentum Difference
    high_vol_momentum = high_vol_days['volume_weighted_momentum'].mean()
    low_vol_momentum = low_vol_days['volume_weighted_momentum'].mean()
    momentum_diff = high_vol_momentum - low_vol_momentum
    
    # Perform Sentiment Analysis
    # Assume we have a sentiment score column in the dataframe
    if 'sentiment_score' in df.columns:
        df['aggregated_sentiment'] = df['sentiment_score'].rolling(window=5).mean()
    else:
        df['aggregated_sentiment'] = 0  # Placeholder for no sentiment data
    
    # Adjust for Overall Market Trend
    market_trend = (df['market_index_close'] / df['market_index_close'].shift(1)) - 1
    df['adjusted_momentum_diff'] = momentum_diff - market_trend
    
    # Incorporate Price Trend
    n = 30  # Example period, can be adjusted
    df['price_trend'] = (df['close'] / df['close'].shift(n)) - 1
    
    # Generate Final Alpha Factor
    df['alpha_factor'] = df['adjusted_momentum_diff'] + df['price_trend'] + df['aggregated_sentiment']
    
    return df['alpha_factor']

# Example usage:
# df = pd.DataFrame(...)  # Load your market data into a DataFrame
# alpha_factor = heuristics_v2(df)
