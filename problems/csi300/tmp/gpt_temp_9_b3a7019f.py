import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate the 5-day and 20-day EMA of Close prices
    ema_close_5 = df['close'].ewm(span=5, adjust=False).mean()
    ema_close_20 = df['close'].ewm(span=20, adjust=False).mean()
    
    # Calculate the 5-day EMA of High prices and 20-day EMA of Low prices
    ema_high_5 = df['high'].ewm(span=5, adjust=False).mean()
    ema_low_20 = df['low'].ewm(span=20, adjust=False).mean()
    
    # Calculate momentum components
    close_momentum = ema_close_5 - ema_close_20
    high_low_momentum = ema_high_5 - ema_low_20
    
    # Combine close, high, and low momentum
    combined_momentum = close_momentum + high_low_momentum
    
    # Measure Volume Impact
    ema_volume_10 = df['volume'].ewm(span=10, adjust=False).mean()
    
    # Adjust for Volume
    volume_ratio = df['volume'] / ema_volume_10
    adjusted_momentum = combined_momentum * volume_ratio
    
    # Calculate Close-to-Low Distance
    close_to_low_distance = df['close'] - df['low']
    
    # Calculate High-to-Low Intraday Range
    high_to_low_range = df['high'] - df['low']
    
    # Calculate High-to-Low Return
    prev_close = df['close'].shift(1)
    high_to_low_return = high_to_low_range / prev_close
    
    # Calculate True Range
    true_range = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close']).abs(),
        (df['low'] - df['close']).abs()
    ], axis=1).max(axis=1)
    
    # Compute Average True Range (ATR)
    atr = true_range.ewm(span=14, adjust=False).mean()
    
    # Adjust Combined Momentum
    adjusted_momentum_by_atr = adjusted_momentum / atr
    
    # Incorporate Enhanced Reversal Sensitivity
    high_low_spread = df['high'] - df['low']
    open_close_spread = df['open'] - df['close']
    
    weighted_high_low_spread = high_low_spread * df['volume']
    weighted_open_close_spread = open_close_spread * df['volume']
    
    combined_weighted_spreads = weighted_high_low_spread + weighted_open_close_spreads
    adjusted_momentum_reversal = adjusted_momentum_by_atr - combined_weighted_spreads
    
    # Combine Factors
    factor = adjusted_momentum_reversal * close_to_low_distance + high_to_low_return
    factor *= df['volume']
    
    # Incorporate Price Gaps
    open_to_close_gap = df['open'] - df['close'].shift(1)
    factor += open_to_close_gap
    
    return factor
