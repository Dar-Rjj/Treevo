import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Daily Price Changes
    df['daily_change'] = df['close'] - df['close'].shift(1)
    
    # Multi-day Price Trends (Exponential Moving Average of daily close price changes)
    df['ema_3_days'] = df['daily_change'].ewm(span=3, adjust=False).mean()
    df['ema_7_days'] = df['daily_change'].ewm(span=7, adjust=False).mean()
    df['ema_21_days'] = df['daily_change'].ewm(span=21, adjust=False).mean()
    
    # Long-Term Momentum Adjusted for Volume
    n = 21  # Lookback period
    df['long_term_momentum'] = (df['close'] - df['close'].shift(n)) / df['close'].shift(n)
    df['volume_ratio'] = df['volume'] / df['volume'].shift(n)
    df['volume_adjusted_momentum'] = df['long_term_momentum'] * df['volume_ratio']
    
    # Multi-day Volume Trends (Exponential Moving Average of daily volume changes)
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    df['volume_ema_3_days'] = df['volume_change'].ewm(span=3, adjust=False).mean()
    df['volume_ema_7_days'] = df['volume_change'].ewm(span=7, adjust=False).mean()
    df['volume_ema_21_days'] = df['volume_change'].ewm(span=21, adjust=False).mean()
    
    # Enhanced Intraday Momentum
    df['high_indicator'] = df['high'] - df['open']
    df['low_indicator'] = df['open'] - df['low']
    df['intraday_momentum'] = (df['high_indicator'] + df['low_indicator']) / 2
    
    # Adjust for Dynamic Volume Impact on Intraday Momentum
    volume_lookback = 21
    df['vol_ma'] = df['volume'].ewm(span=volume_lookback, adjust=False).mean()
    df['normalized_volume'] = df['volume'] / df['vol_ma']
    df['adjusted_intraday_momentum'] = df['intraday_momentum'] * df['normalized_volume']
    
    # Gaps and Breaks
    df['up_gap'] = (df['open'] - df['close'].shift(1)).clip(lower=0)
    df['down_gap'] = (df['open'] - df['close'].shift(1)).clip(upper=0)
    
    max_high_21 = df['high'].rolling(window=21).max()
    min_low_21 = df['low'].rolling(window=21).min()
    
    df['upper_breakout'] = (df['high'] > max_high_21).astype(float) * (df['high'] - max_high_21)
    df['lower_breakout'] = (df['low'] < min_low_21).astype(float) * (min_low_21 - df['low'])
    
    # Intraday Patterns
    df['intraday_range'] = df['high'] - df['low']
    df['intraday_momentum_close_open'] = df['close'] - df['open']
    df['intraday_strength'] = df['intraday_momentum_close_open'] / df['intraday_range']
    
    # Intraday Volatility
    df['volatility'] = df[['high', 'low']].stack().ewm(span=21, adjust=False).std(level=0)
    
    # Volume Spike Filter
    spike_threshold = 5
    df['volume_spike'] = df['volume'] / df['volume'].shift(1)
    df['no_spike'] = (df['volume_spike'] <= spike_threshold).astype(float)
    
    # Consolidate All Factors
    df['weighted_sum'] = (
        0.4 * (df['ema_3_days'] + df['ema_7_days'] + df['ema_21_days']) +
        0.3 * (df['volume_adjusted_momentum'] + df['adjusted_intraday_momentum']) +
        0.3 * (df['up_gap'] - df['down_gap'] + df['upper_breakout'] - df['lower_breakout']) * df['no_spike']
    )
    
    # Final Alpha Factor
    df['alpha_factor'] = df['weighted_sum'] / df['open']
    
    return df['alpha_factor'].dropna()
