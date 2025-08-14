import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Range Indicator
    df['intraday_range'] = df['high'] - df['low']
    
    # Integrate Volume Weighted Average Price (VWAP)
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['vwap_diff'] = df['close'] - df['vwap']
    
    # Combine Indicators for Intraday Analysis
    df['intraday_indicator'] = df['intraday_range'] + df['vwap_diff']
    
    # Calculate Adjusted Volume
    df['adjusted_volume'] = df['volume'] / df['intraday_range']
    
    # Integrate Volume-Adjusted Log Return
    df['log_return'] = df['close'].apply(lambda x: np.log(x) - np.log(df.loc[df.index.get_loc(x) - 1, 'close']))
    df['volume_adjusted_log_return'] = df['log_return'] * df['adjusted_volume']
    
    # Calculate Daily Price Change
    df['daily_price_change'] = df['close'].diff()
    
    # Compute Volume-Averaged Daily Return
    N = 10
    df['volume_weighted_daily_return'] = (df['daily_price_change'] * df['volume']).rolling(window=N).sum() / df['volume'].rolling(window=N).sum()
    
    # Calculate Raw Momentum
    df['raw_momentum'] = df['close'] - df['close'].shift(7)
    
    # Incorporate Volume Change
    df['volume_factor'] = df['volume'] / df['volume'].rolling(window=7).mean()
    
    # Adjust for Volatility
    df['short_term_volatility'] = df['close'].rolling(window=7).std()
    df['adjusted_momentum'] = df['raw_momentum'] / df['short_term_volatility'] * df['volume_factor']
    
    # Combine Factors
    df['intermediate_alpha_factor'] = df['volume_adjusted_log_return'] * df['adjusted_momentum']
    
    # Confirm with Volume and Amount Trends
    M = 10
    df['avg_volume'] = df['volume'].rolling(window=M).mean()
    df['volume_trend_score'] = df.apply(lambda x: 1 if x['volume'] > x['avg_volume'] else -1, axis=1)
    
    K = 10
    df['avg_amount'] = df['amount'].rolling(window=K).mean()
    df['amount_trend_score'] = df.apply(lambda x: 1 if x['amount'] > x['avg_amount'] else -1, axis=1)
    
    # Calculate Long-Term (LMA) and Short-Term (SMA) Moving Averages
    LMA_period = 200
    SMA_period = 50
    df['lma'] = df['close'].rolling(window=LMA_period).mean()
    df['sma'] = df['close'].rolling(window=SMA_period).mean()
    
    # Calculate Difference Between LMA and SMA
    df['lma_sma_diff'] = df['lma'] - df['sma']
    
    # Apply Volume Weighted Filter
    df['volume_weighted_lma_sma_diff'] = df['lma_sma_diff'] * df['volume']
    df['smoothed_volume_weighted_lma_sma_diff'] = df['volume_weighted_lma_sma_diff'].ewm(span=10).mean()
    
    # Final Alpha Factor
    df['final_alpha_factor'] = df['intermediate_alpha_factor'] * df['volume_weighted_daily_return']
    
    return df['final_alpha_factor']
