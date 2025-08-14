import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Volume-Weighted Average Price (VWAP)
    df['vwap'] = ((df['high'] + df['low']) / 2 * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Calculate Short-Term EMA of Close Prices (e.g., 5 days)
    short_span = 5
    df['ema_short'] = df['close'].ewm(span=short_span, adjust=False).mean()
    
    # Calculate Long-Term EMA of Close Prices (e.g., 20 days)
    long_span = 20
    df['ema_long'] = df['close'].ewm(span=long_span, adjust=False).mean()
    
    # Generate the Momentum Factor
    df['momentum_factor'] = df['ema_short'] - df['ema_long']
    
    # Calculate Daily Return Using VWAP
    df['daily_return_vwap'] = (df['close'] - df['vwap']) / df['vwap']
    
    # Smooth and Scale the Daily Return
    ema_daily_return_span = 10
    df['smoothed_daily_return'] = df['daily_return_vwap'].ewm(span=ema_daily_return_span, adjust=False).mean()
    df['scaled_daily_return'] = df['smoothed_daily_return'] * df['volume']
    
    # Calculate High-to-Low Range
    df['range'] = df['high'] - df['low']
    
    # Calculate Open-Adjusted Range
    df['open_adjusted_range'] = df[['high' - 'open', 'open' - 'low']].max(axis=1)
    
    # Calculate Enhanced Price Momentum with Volume Adjustment
    lookback_period = 15
    df['ema_close_15'] = df['close'].ewm(span=lookback_period, adjust=False).mean()
    df['price_diff'] = df['close'] - df['ema_close_15']
    df['momentum_score'] = df['price_diff'] / df['ema_close_15']
    df['cumulative_volume'] = df['volume'].rolling(window=lookback_period).sum()
    df['adjusted_momentum_score'] = df['momentum_score'] * df['cumulative_volume']
    
    # Calculate Trading Intensity
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    df['amount_change'] = df['amount'] - df['amount'].shift(1)
    df['trading_intensity'] = df['volume_change'] / df['amount_change']
    
    # Weight the Range by Trading Intensity
    intensity_scale = 1800
    df['weighted_range'] = (intensity_scale * df['trading_intensity']) * df['range']
    
    # Calculate Volume-Weighted VWAP
    df['vwap_weighted'] = (df['vwap'] * df['volume']).rolling(window=lookback_period).sum() / df['volume'].rolling(window=lookback_period).sum()
    
    # Combine All Alpha Factors
    df['alpha_factor'] = (
        df['momentum_factor'] +
        df['scaled_daily_return'] +
        df['open_adjusted_range'] +
        df['adjusted_momentum_score'] +
        df['weighted_range'] +
        df['vwap_weighted']
    )
    
    return df['alpha_factor']
