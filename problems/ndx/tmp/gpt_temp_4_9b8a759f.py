import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Volume-Weighted Average Price (VWAP)
    df['vwap'] = (df['high'] + df['low']) / 2 * df['volume']
    df['vwap'] = df['vwap'].cumsum() / df['volume'].cumsum()
    
    # Calculate Daily Return Using VWAP
    df['daily_return'] = (df['close'] - df['vwap']) / df['vwap']
    
    # Smooth and Scale the Daily Return
    span = 10
    df['smoothed_return'] = df['daily_return'].ewm(span=span, adjust=False).mean()
    df['scaled_return'] = df['smoothed_return'] * df['volume']
    
    # Calculate High-to-Low Range
    df['range'] = df['high'] - df['low']
    
    # Calculate Open-Adjusted Range
    df['open_adjusted_range'] = df[['high', 'low']].apply(lambda x: max(x[0] - df['open'], df['open'] - x[1]), axis=1)
    
    # Calculate Enhanced Price Momentum with Volume Adjustment
    lookback_period = 10
    df['ema_close'] = df['close'].ewm(span=lookback_period, adjust=False).mean()
    df['price_diff'] = df['close'] - df['ema_close']
    df['momentum_score'] = df['price_diff'] / df['ema_close']
    df['cumulative_volume'] = df['volume'].rolling(window=lookback_period, min_periods=1).sum()
    df['adjusted_momentum'] = df['momentum_score'] * df['cumulative_volume']
    
    # Calculate Trading Intensity
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    df['amount_change'] = df['amount'] - df['amount'].shift(1)
    df['trading_intensity'] = df['volume_change'] / df['amount_change']
    
    # Weight the Range by Trading Intensity
    scale_factor = 1500
    df['weighted_range'] = df['trading_intensity'] * scale_factor * df['range']
    
    # Combine Momentum and Weighted Range
    df['combined_factor'] = (
        df['scaled_return'] +
        df['open_adjusted_range'] +
        df['adjusted_momentum'] +
        df['weighted_range']
    )
    
    # Calculate Volume-Weighted Momentum
    n_days = 10
    df['product_daily_return_volume'] = df['daily_return'] * df['volume']
    df['aggregate_product'] = df['product_daily_return_volume'].rolling(window=n_days, min_periods=1).sum()
    df['aggregate_volume'] = df['volume'].rolling(window=n_days, min_periods=1).sum()
    df['vwap_momentum'] = df['aggregate_product'] / df['aggregate_volume']
    
    # Final VWAM Calculation
    df['final_vwam'] = df['vwap_momentum'] * df['scaled_return']
    
    # Combine All Alpha Factors
    df['alpha_factor'] = (
        df['combined_factor'] +
        df['final_vwam']
    )
    
    return df['alpha_factor']
