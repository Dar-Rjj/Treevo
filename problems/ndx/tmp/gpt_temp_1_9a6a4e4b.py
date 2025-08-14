import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate EMAs
    df['5_day_EMA'] = df['close'].ewm(span=5, adjust=False).mean()
    df['20_day_EMA'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # Momentum Difference
    df['momentum_diff'] = df['20_day_EMA'] - df['5_day_EMA']
    
    # Volatility Adjustment
    df['daily_range'] = df['high'] - df['low']
    df['30_day_avg_daily_range'] = df['daily_range'].rolling(window=30).mean()
    
    # Volume Influence
    df['30_day_avg_volume'] = df['volume'].rolling(window=30).mean()
    df['volume_deviation_score'] = ((df['volume'] - df['30_day_avg_volume']) / df['30_day_avg_volume']) * 100
    
    # Intraday Measures
    df['intraday_high_low_spread'] = df['high'] - df['low']
    df['close_open_diff'] = df['close'] - df['open']
    df['combined_intraday_measure'] = (df['intraday_high_low_spread'] + df['close_open_diff']) * df['volume']
    
    # Short-Term Price Momentum
    df['daily_price_change'] = df['close'].diff()
    df['5_day_ema_price_change'] = df['daily_price_change'].ewm(span=5, adjust=False).mean()
    df['10_day_ema_price_change'] = df['daily_price_change'].ewm(span=10, adjust=False).mean()
    df['short_term_momentum'] = df['5_day_ema_price_change'] - df['10_day_ema_price_change']
    
    # Volume Indicator
    df['volume_change'] = df['volume'].diff()
    df['positive_volume_days'] = (df['volume_change'] > 0).rolling(window=20).sum()
    df['negative_volume_days'] = (df['volume_change'] < 0).rolling(window=20).sum()
    
    # Average True Range (ATR)
    df['true_range'] = df.apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - df.shift(1).loc[x.name, 'close']), abs(x['low'] - df.shift(1).loc[x.name, 'close'])), axis=1)
    df['14_day_sma_true_range'] = df['true_range'].rolling(window=14).mean()
    
    # Combine Short-Term and Long-Term Price Momentum with Volume Indicator
    df['short_term_combined'] = df['short_term_momentum'] * df['positive_volume_days'] / df['14_day_sma_true_range']
    df['long_term_combined'] = (df['momentum_diff'] * df['negative_volume_days']) / df['14_day_sma_true_range']
    
    # Final Factor Adjustment
    df['final_factor_adjustment'] = (df['short_term_momentum'] * df['volume_deviation_score']) / df['combined_intraday_measure']
    
    # High-Low Range Factor
    df['high_low_range_factor'] = (df['high'] - df['low']) / df['close']
    
    # Combine Momentum, Volume, and High-Low Range
    df['combined_factors'] = (df['short_term_momentum'] * df['volume_deviation_score'] * df['high_low_range_factor']) / df['close']
    
    # Combine Factors
    df['short_term_vol_adj_range'] = df['5_day_ema_price_change'].rolling(window=5).mean() * df['volume_deviation_score']
    df['long_term_vol_adj_range'] = df['10_day_ema_price_change'].rolling(window=20).mean() * df['volume_deviation_score']
    df['adjusted_momentum'] = df['momentum_diff'] / df['30_day_avg_daily_range']
    
    # Final Alpha Factor
    df['final_alpha_factor'] = (df['short_term_vol_adj_range'] - df['long_term_vol_adj_range'] + 
                                df['final_factor_adjustment'] + df['adjusted_momentum']) * df['volume_deviation_score']
    
    return df['final_alpha_factor']
