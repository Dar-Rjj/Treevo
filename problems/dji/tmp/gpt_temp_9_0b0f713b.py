import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Exponential Daily Returns
    df['daily_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['exp_daily_return'] = df['daily_return'].ewm(span=5, adjust=False).mean()
    
    # Short-Term and Long-Term EMA of Daily Returns
    df['short_term_ema'] = df['daily_return'].ewm(span=5, adjust=False).mean()
    df['long_term_ema'] = df['daily_return'].ewm(span=20, adjust=False).mean()
    
    # Dynamic Difference
    df['dynamic_diff'] = df['short_term_ema'] - df['long_term_ema']
    
    # Volume-Weighted Momentum Indicators
    df['weighted_daily_return'] = df['daily_return'] * df['volume']
    df['weighted_5d_ma'] = df['weighted_daily_return'].rolling(window=5).mean()
    
    # Short-Term and Long-Term EMA of Volume
    df['short_term_ema_volume'] = df['volume'].ewm(span=5, adjust=False).mean()
    df['long_term_ema_volume'] = df['volume'].ewm(span=20, adjust=False).mean()
    
    # Volume Momentum
    df['volume_momentum'] = df['short_term_ema_volume'] - df['long_term_ema_volume']
    
    # 10-day MA of High-Low Difference
    df['high_low_diff'] = df['high'] - df['low']
    df['ma_high_low_diff'] = df['high_low_diff'].rolling(window=10).mean()
    
    # 10-day Volume-Weighted MA of Open-Close Difference
    df['open_close_diff'] = (df['open'] - df['close']) * df['volume']
    df['ma_open_close_diff'] = df['open_close_diff'].rolling(window=10).mean() / df['volume'].rolling(window=10).mean()
    
    # Cumulative Return with Adjustments
    N = 10
    df['cumulative_return'] = (df['close'] / df['close'].shift(N) - 1)
    df['volume_influence'] = df['cumulative_return'] * df['volume'].rolling(window=N).mean()
    df['price_range_adjustment'] = df['volume_influence'] / (df['high'].rolling(window=N).max() - df['low'].rolling(window=N).min())
    
    # Adjusted High-Low Spread with True Range
    df['true_range'] = df[['high' - 'low', 'high' - df['close'].shift(1), df['close'].shift(1) - 'low']].max(axis=1)
    df['adjusted_high_low_spread'] = df['high'] - df['low'] + df['true_range']
    df['atr'] = df['true_range'].rolling(window=14).mean()
    df['volume_weighted_spread_atr'] = (df['adjusted_high_low_spread'] * df['volume']) / df['atr']
    
    # Condition on Close-to-Open Return
    df['close_to_open_return'] = (df['close'] - df['open']) / df['open']
    df['return_weight'] = np.where(df['close_to_open_return'] > 0, 1, -1)
    
    # Intraday Percent Change
    df['intraday_percent_change'] = (df['close'] - df['open']) / df['open']
    
    # Momentum using ATR
    n_periods = 10
    df['momentum_atr'] = (df['close'] - df['close'].shift(n_periods)) / df['atr']
    
    # Combined Alpha Factor
    df['alpha_factor'] = (
        (df['dynamic_diff'] + df['volume_momentum']) / 2 +
        df['price_range_adjustment'] +
        df['volume_weighted_spread_atr'] +
        df['intraday_percent_change'] +
        df['momentum_atr']
    )
    
    return df['alpha_factor']
