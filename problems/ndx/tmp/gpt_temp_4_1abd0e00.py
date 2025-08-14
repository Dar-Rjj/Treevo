import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Daily Log Returns
    df['log_returns'] = np.log(df['close']) - np.log(df['close'].shift(1))
    
    # 20-Day EMA of Log Returns, Weigh by Volume
    df['volume_weighted_returns'] = df['log_returns'] * df['volume']
    df['ema_volume_weighted_returns'] = df['volume_weighted_returns'].ewm(span=20).mean()
    
    # High-Low Range Volatility
    df['high_low_range'] = df['high'] - df['low']
    df['range_volatility'] = df['high_low_range'].rolling(window=20).std()
    df['volume_adjusted_price_movement'] = df['ema_volume_weighted_returns'] / df['range_volatility']
    
    # 10-day Simple Moving Average (SMA) of the Close Price
    df['sma_close_10'] = df['close'].rolling(window=10).mean()
    
    # Momentum Factor
    df['momentum_factor'] = df['close'] - df['sma_close_10']
    df['momentum_sign'] = np.sign(df['momentum_factor'])
    
    # Identify High-Activity Days
    df['volume_sma_30'] = df['volume'].rolling(window=30).mean()
    df['is_high_activity'] = df['volume'] > df['volume_sma_30']
    
    # Percentage Change in Close Price on High-Activity Days
    df['close_pct_change'] = df['close'].pct_change()
    df['high_activity_momentum'] = df['close_pct_change'] * df['is_high_activity'].astype(int)
    df['adjusted_momentum_score'] = df['momentum_factor'] + df['high_activity_momentumscore']
    
    # Analyze Intraday Market Sentiment
    df['daily_midpoint'] = (df['high'] + df['low']) / 2
    df['directional_bias'] = np.where(df['close'] > df['daily_midpoint'], 1, -1)
    
    # Composite Signal
    df['composite_signal'] = df['momentum_factor'] * df['directional_bias']
    
    # Filter by Volume
    df['composite_signal_filtered'] = df['composite_signal'] * (df['volume'] > df['volume_sma_30']).astype(int)
    
    # Analyze Price Reversals Using High, Low, and Open Prices
    df['price_reversal'] = (df['high'] - df['low']) / df['open']
    df['sum_price_reversal_20'] = df['price_reversal'].rolling(window=20).sum()
    
    # Assess Volume Impact on Price Movement
    df['volume_weighted_close'] = df['close'] * df['volume']
    df['ewm_volume_weighted_close'] = df['volume_weighted_close'].ewm(span=20).mean()
    
    # Incorporate Trend Following with Moving Averages
    df['sma_close_50'] = df['close'].rolling(window=50).mean()
    df['sma_close_200'] = df['close'].rolling(window=200).mean()
    df['trend_direction'] = np.where(df['sma_close_50'] > df['sma_close_200'], 1, -1)
    
    # Combine Factors for Final Alpha
    df['final_alpha'] = (
        0.4 * df['volume_adjusted_price_movement'] +
        0.1 * df['composite_signal_filtered'] +
        0.3 * df['sum_price_reversal_20'] +
        0.1 * df['ewm_volume_weighted_close'] +
        0.05 * df['composite_signal'] +
        0.05 * df['trend_direction']
    )
    
    return df['final_alpha']
