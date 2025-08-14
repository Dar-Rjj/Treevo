import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Gaps
    df['daily_gap'] = df['open'] - df['close'].shift(1)
    
    # Calculate Volume Weighted Average of Gaps
    df['gap_volume_weighted'] = (df['daily_gap'] * df['volume']).rolling(window=10).sum() / df['volume'].rolling(window=10).sum()
    
    # Calculate Simple Momentum
    n = 5
    df['simple_momentum'] = df['close'] - df['close'].shift(n)
    
    # Volume Adjusted Component
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    df['volume_adjusted_momentum'] = df['simple_momentum'] * (df['volume_change'] / df['volume'])
    
    # Enhanced Price Reversal Sensitivity
    df['high_low_spread'] = df['high'] - df['low']
    df['open_close_spread'] = df['open'] - df['close']
    df['weighted_high_low_spread'] = df['high_low_spread'] * (df['volume_change'] / df['volume'])
    df['weighted_open_close_spread'] = df['open_close_spread'] * (df['volume_change'] / df['volume'])
    df['combined_weighted_spreads'] = df['weighted_high_low_spread'] + df['weighted_open_close_spread']
    
    # Volume Trend Component
    df['ema_volume'] = df['volume'].ewm(span=10, adjust=False).mean()
    df['volume_trend'] = df['volume'] - df['ema_volume']
    
    # Price Trend Component
    df['ema_price'] = df['close'].ewm(span=10, adjust=False).mean()
    df['price_trend'] = df['close'] - df['ema_price']
    
    # Incorporate Enhanced Gaps
    df['open_to_close_gap'] = df['open'] - df['close']
    df['high_to_low_gap'] = df['high'] - df['low']
    df['enhanced_gaps'] = df['open_to_close_gap'] + df['high_to_low_gap']
    
    # Calculate Volume Surge
    avg_volume = df['volume'].rolling(window=10).mean()
    df['volume_surge'] = (df['volume'] > 1.5 * avg_volume).astype(int)
    
    # Measure Volume Impact
    df['ema_volume_10'] = df['volume'].ewm(span=10, adjust=False).mean()
    
    # Introduce Volatility Component
    df['daily_volatility'] = df[['high', 'low']].apply(lambda x: max(abs(x[0] - x[1]), abs(x[0] - df['close'].shift(1)), abs(x[1] - df['close'].shift(1))), axis=1)
    df['ema_volatility_10'] = df['daily_volatility'].ewm(span=10, adjust=False).mean()
    
    # Confirm with ATR
    df['true_range'] = df[['high', 'low', 'close']].apply(lambda x: max(abs(x[0] - x[1]), abs(x[0] - x[2].shift(1)), abs(x[1] - x[2].shift(1))), axis=1)
    df['atr_14'] = df['true_range'].rolling(window=14).mean()
    
    # Combine Components
    df['final_factor'] = (
        df['volume_adjusted_momentum'] -
        df['combined_weighted_spreads'] +
        df['volume_trend'] +
        df['price_trend'] +
        df['enhanced_gaps']
    ) * df['volume_surge'] / df['ema_volume_10'] * df['ema_volatility_10'] / df['atr_14']
    
    return df['final_factor']
