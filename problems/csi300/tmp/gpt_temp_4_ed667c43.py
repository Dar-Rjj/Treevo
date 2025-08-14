import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Price Movement
    df['high_low_range'] = df['high'] - df['low']
    df['close_open_diff'] = df['close'] - df['open']
    df['intraday_movement'] = (df['high_low_range'] + df['close_open_diff']) / 2

    # Incorporate Volume Influence
    df['volume_adjusted_momentum'] = df['volume'] * df['intraday_movement']

    # Adaptive Smoothing via Moving Average
    df['daily_return'] = df['close'].pct_change()
    df['abs_daily_return'] = df['daily_return'].abs()
    df['robust_volatility'] = df['abs_daily_return'].rolling(window=30).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    df['dynamic_ema_period'] = (df['robust_volatility'] / df['abs_daily_return']).fillna(10).round().astype(int).clip(lower=5, upper=20)
    
    def dynamic_ema(series, periods):
        return series.ewm(span=periods, adjust=False).mean()
    
    df['smoothed_volume_adj_momentum'] = df.groupby('dynamic_ema_period')['volume_adjusted_momentum'].transform(dynamic_ema)

    # Adjust for Market Volatility
    df['adjusted_volume_adj_momentum'] = df['smoothed_volume_adj_momentum'] / df['robust_volatility']

    # Incorporate Trend Reversal Signal
    df['short_term_momentum'] = df['close'].ewm(span=5, adjust=False).mean()
    df['long_term_momentum'] = df['close'].ewm(span=20, adjust=False).mean()
    df['momentum_reversal'] = df['short_term_momentum'] - df['long_term_momentum']
    df['reversal_signal'] = np.sign(df['momentum_reversal'])

    # Integrate Non-Linear Transformation
    df['sqrt_transformed_momentum'] = np.sqrt(np.abs(df['adjusted_volume_adj_momentum']))
    df['log_transformed_momentum'] = np.log1p(np.abs(df['adjusted_volume_adj_momentum']))

    # Enhance Reversal Signal with Adaptive Smoothing
    df['smoothed_reversal_signal'] = df.groupby('dynamic_ema_period')['reversal_signal'].transform(dynamic_ema)

    # Combine Smoothed Reversal Signal with Non-Linearly Transformed Momentum
    df['interim_alpha_factor'] = df['sqrt_transformed_momentum'] + df['log_transformed_momentum'] + df['smoothed_reversal_signal']

    # Final Adaptive Smoothing
    df['final_alpha_factor'] = df.groupby('dynamic_ema_period')['interim_alpha_factor'].transform(dynamic_ema)

    return df['final_alpha_factor']
