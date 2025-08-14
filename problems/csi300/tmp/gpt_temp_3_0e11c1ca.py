import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Return
    df['daily_return'] = df['close'].pct_change()
    
    # Compute Exponential Moving Average (EMA) of Returns
    ema_period = 10
    df['ema_returns'] = df['daily_return'].ewm(span=ema_period, adjust=False).mean()
    
    # Calculate True Range
    df['true_range'] = df[['high', 'low']].max(axis=1) - df[['high', 'low']].min(axis=1)
    
    # Calculate Average True Range (ATR)
    atr_period = 14
    df['atr'] = df['true_range'].rolling(window=atr_period).mean()
    
    # Categorize Volatility Days
    volatility_threshold = df['atr'].quantile(0.75)
    df['volatility'] = df['atr'].apply(lambda x: 'high' if x > volatility_threshold else 'low')
    
    # Filter Days by Volatility
    high_vol_days = df[df['volatility'] == 'high']
    low_vol_days = df[df['volatility'] == 'low']
    
    # Calculate Volume Weighted Momentum
    df['volume_weighted_momentum'] = df['daily_return'] * df['volume']
    
    # Compute Momentum Difference
    high_vol_momentum = high_vol_days['volume_weighted_momentum'].mean()
    low_vol_momentum = low_vol_days['volume_weighted_momentum'].mean()
    momentum_diff = high_vol_momentum - low_vol_momentum
    
    # Perform Sentiment Analysis (Assuming sentiment scores are available in a column 'sentiment_score')
    df['sentiment_score'] = df['sentiment_score'].fillna(0)
    
    # Adjust for Overall Market Trend
    df['market_trend'] = df['market_index_close'].pct_change()
    df['adjusted_market_trend'] = df['daily_return'] - df['market_trend']
    
    # Incorporate Price Trend
    price_trend_period = 30
    df['price_trend'] = df['close'].pct_change(periods=price_trend_period)
    
    # Generate Final Alpha Factor
    df['alpha_factor'] = (momentum_diff + df['adjusted_market_trend'] + 
                          df['price_trend'] + df['sentiment_score'])
    
    return df['alpha_factor']
