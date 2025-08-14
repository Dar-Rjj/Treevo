import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Define lookback periods and weights
    sma_lookback = 20
    vol_lookback = 30
    momentum_lookback = 10
    pct_change_lookback = 5
    high_low_std_lookback = 20
    trend_lookback = 60
    ema_lookback = 14
    rsi_lookback = 14
    bullish_weight = 1.2
    bearish_weight = 0.8
    sideways_weight = 1.0
    ema_weight = 0.6
    rsi_weight = 0.4
    momentum_weight = 0.5
    vol_weight = 0.3
    additional_metrics_weight = 0.2
    
    # Calculate Simple Moving Average (SMA) of Close Prices
    df['SMA'] = df['close'].rolling(window=sma_lookback).mean()
    
    # Compute Volume-Adjusted Volatility
    df['high_low_diff'] = df['high'] - df['low']
    df['vol_adjusted_vol'] = df['high_low_diff'] * df['volume']
    df['vol_adjusted_vol_rolling_avg'] = df['vol_adjusted_vol'].rolling(window=vol_lookback).mean()
    
    # Compute Price Momentum
    df['price_momentum'] = (df['close'] - df['SMA']) / df['close'].rolling(window=momentum_lookback).mean()
    
    # Incorporate Additional Price Change Metrics
    df['pct_change_close'] = df['close'].pct_change(periods=pct_change_lookback)
    df['high_low_range'] = df['high'] - df['low']
    df['high_low_range_std'] = df['high_low_range'].rolling(window=high_low_std_lookback).std()
    
    # Incorporate Market Conditions
    def determine_market_trend(series):
        slope, _, _, _, _ = linregress(range(len(series)), series)
        if slope > 0.05:
            return 'bullish'
        elif slope < -0.05:
            return 'bearish'
        else:
            return 'sideways'
    
    df['market_trend'] = df['close'].rolling(window=trend_lookback).apply(determine_market_trend, raw=False)
    
    def apply_market_weights(row):
        if row['market_trend'] == 'bullish':
            return bullish_weight
        elif row['market_trend'] == 'bearish':
            return bearish_weight
        else:
            return sideways_weight
    
    df['market_weight'] = df.apply(apply_market_weights, axis=1)
    
    # Integrate Advanced Volatility Measures
    df['ema_high_low'] = df['high_low_diff'].ewm(span=ema_lookback, adjust=False).mean()
    
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_lookback).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_lookback).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['advanced_volatility'] = (ema_weight * df['ema_high_low']) + (rsi_weight * df['rsi'])
    
    # Final Alpha Factor
    df['alpha_factor'] = (
        (momentum_weight * df['price_momentum']) +
        (vol_weight * df['vol_adjusted_vol_rolling_avg']) +
        (additional_metrics_weight * df['pct_change_close'] * df['high_low_range_std']) +
        df['market_weight'] * df['advanced_volatility']
    )
    
    return df['alpha_factor']
