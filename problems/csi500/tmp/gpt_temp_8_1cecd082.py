import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Short-Term Price Momentum
    short_term_mom = df['close'] - df['close'].rolling(window=10).mean()
    
    # Calculate Medium-Term Price Momentum
    medium_term_mom = df['close'] - df['close'].rolling(window=30).mean()
    
    # Calculate Long-Term Price Momentum
    long_term_mom = df['close'] - df['close'].rolling(window=50).mean()
    
    # Combine Multi-Period Momenta
    combined_momentum = short_term_mom + medium_term_mom + long_term_mom
    
    # Calculate Volume-Weighted Average Return
    daily_returns = (df['close'] / df['open']) - 1
    volume_weighted_returns = (daily_returns * df['volume']).sum() / df['volume'].sum()
    
    # Adjust Combined Momentum by Volume-Weighted Average Return
    adjusted_momentum = combined_momentum * volume_weighted_returns
    
    # Assess Trend Following Potential
    trend_weight = 1 if df['close'].rolling(window=50).mean().iloc[-1] > df['close'].iloc[-1] else 0.5
    trend_following_component = adjusted_momentum * trend_weight
    
    # Determine Intermediate Factor Value
    intermediate_factor_value = adjusted_momentum + trend_following_component
    
    # Calculate Short-Term Volatility
    short_term_vol = ((df['high'] - df['low']).ewm(span=10).mean()).fillna(0)
    
    # Calculate Medium-Term Volatility
    medium_term_vol = ((df['high'] - df['low']).ewm(span=30).mean()).fillna(0)
    
    # Calculate Long-Term Volatility
    long_term_vol = ((df['high'] - df['low']).ewm(span=50).mean()).fillna(0)
    
    # Combine Multi-Period Volatilities
    combined_volatility = short_term_vol + medium_term_vol + long_term_vol
    
    # Adjust Combined Momentum by Combined Volatility
    volatility_adjusted_momentum = intermediate_factor_value / combined_volatility
    
    # Incorporate Market Sentiment (Assume a column 'sentiment' in the dataframe)
    sentiment_ema = df['sentiment'].ewm(span=50).mean()
    sentiment_adjusted_momentum = volatility_adjusted_momentum * sentiment_ema
    
    # Incorporate Liquidity Measures
    turnover = df['volume'] / df['close']
    turnover_ema = turnover.ewm(span=50).mean()
    liquidity_adjusted_momentum = sentiment_adjusted_momentum * turnover_ema
    
    # Incorporate Seasonality (Assume a column 'monthly_performance' in the dataframe)
    monthly_seasonal_pattern = df['monthly_performance'].ewm(span=12).mean()
    final_factor_value = liquidity_adjusted_momentum * monthly_seasonal_pattern
    
    return final_factor_value
