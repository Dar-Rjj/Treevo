import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Adaptive Window Calculation
    df['volatility'] = df[['high', 'low', 'close']].rolling(window=30).std().mean(axis=1)
    df['window_size'] = (30 - (df['volatility'] - df['volatility'].min()) / (df['volatility'].max() - df['volatility'].min()) * 10).astype(int)
    
    # Momentum Factor
    short_term_sma = df['close'].rolling(window=10).mean()
    long_term_sma = df['close'].rolling(window=50).mean()
    df['momentum_factor'] = short_term_sma - long_term_sma
    
    # Liquidity Measure
    average_volume = df['volume'].rolling(window=30).mean()
    df['liquidity_measure'] = df['volume'] / average_volume
    
    # Trend Strength
    df['DM_pos'] = df['high'].diff().apply(lambda x: max(x, 0))
    df['DM_neg'] = df['low'].diff().apply(lambda x: max(-x, 0))
    df['DI_pos'] = df['DM_pos'].rolling(window=14).sum() / df['ATR_14']
    df['DI_neg'] = df['DM_neg'].rolling(window=14).sum() / df['ATR_14']
    df['trend_strength'] = df['DI_pos'] - df['DI_neg']
    
    # Market Sentiment
    df['daily_return'] = df['close'].pct_change()
    df['positive_returns'] = df['daily_return'].apply(lambda x: max(x, 0)).rolling(window=30).sum()
    df['negative_returns'] = df['daily_return'].apply(lambda x: min(x, 0)).abs().rolling(window=30).sum()
    df['market_sentiment'] = df['positive_returns'] - df['negative_returns']
    
    # Rolling Statistics
    def rolling_stats(series, window):
        return series.rolling(window=window).mean(), series.rolling(window=window).std()
    
    df['rolling_mean'], df['rolling_std'] = zip(*df.groupby('window_size')['volume_weighted_return'].transform(lambda x: rolling_stats(x, x.name)))
    
    # Final Alpha Factor
    df['alpha_factor'] = (df['volume_weighted_return'] - df['rolling_mean']) / df['rolling_std']
    
    return df['alpha_factor']
