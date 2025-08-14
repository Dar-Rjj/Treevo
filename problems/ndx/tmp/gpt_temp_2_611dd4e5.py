import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Log Returns
    df['log_return'] = df['close'].apply(lambda x: np.log(x / df['close'].shift(1)))
    
    # 20-Day EMA of Log Returns, Weigh by Volume
    df['volume_weighted_log_return'] = df['log_return'] * df['volume']
    df['ema_20_vol_weighted_log_return'] = df['volume_weighted_log_return'].ewm(span=20).mean()
    
    # Daily High-Low Range Volatility
    df['high_low_range'] = df['high'] - df['low']
    df['sd_20_high_low_range'] = df['high_low_range'].rolling(window=20).std()
    df['high_low_range_volatility'] = df['high_low_range'] / df['sd_20_high_low_range']
    
    # Weight by High-Low Range Volatility
    df['vol_adjusted_price_movement'] = df['ema_20_vol_weighted_log_return'] * df['high_low_range_volatility']
    
    # 10-day Simple Moving Average (SMA) of the Close Price
    df['sma_10_close'] = df['close'].rolling(window=10).mean()
    
    # Define a Momentum Factor
    df['momentum_factor'] = df['close'] - df['sma_10_close']
    
    # Identify High-Activity Days
    df['volume_30_sma'] = df['volume'].rolling(window=30).mean()
    df['is_high_activity_day'] = df['volume'] > df['volume_30_sma']
    
    # Calculate Percentage Change in Close Price on High-Activity Days
    df['close_change_high_activity'] = df.apply(lambda x: (x['close'] - df['close'].shift(1)) / df['close'].shift(1) if x['is_high_activity_day'] else 0, axis=1)
    df['adjusted_momentum_score'] = df['momentum_factor'] + df['close_change_high_activity']
    
    # Analyze Intraday Market Sentiment
    df['midpoint'] = (df['high'] + df['low']) / 2
    df['directional_bias'] = df.apply(lambda x: 1 if x['close'] > x['midpoint'] else (-1 if x['close'] < x['midpoint'] else 0), axis=1)
    
    # Combine Momentum and Sentiment for Composite Signal
    df['composite_signal'] = df['momentum_factor'] * df['directional_bias']
    
    # Filter by Volume
    df['filtered_composite_signal'] = df.apply(lambda x: x['composite_signal'] if x['volume'] > df['volume_30_sma'] else 0, axis=1)
    
    # Analyze Price Reversals Using High, Low, and Open Prices
    df['price_reversal'] = (df['high'] - df['low']) / df['open']
    df['sum_20_price_reversal'] = df['price_reversal'].rolling(window=20).sum()
    
    # Assess Volume Impact on Price Movement
    df['volume_weighted_close'] = df['close'] * df['volume']
    df['ema_20_volume_weighted_close'] = df['volume_weighted_close'].ewm(span=20).mean()
    
    # Combine Factors for Final Alpha
    df['alpha'] = (
        0.4 * df['vol_adjusted_price_movement'] +
        0.1 * df['sum_20_price_reversal'] +
        0.3 * df['sum_20_price_reversal'] +
        0.1 * df['ema_20_volume_weighted_close'] +
        0.1 * df['filtered_composite_signal']
    )
    
    return df['alpha']
