import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Gaps
    df['daily_gap'] = df['open'] - df['close'].shift(1)
    
    # Calculate Volume Weighted Average of Gaps
    df['gap_volume_product'] = df['daily_gap'] * df['volume']
    total_gap_volume_product = df['gap_volume_product'].rolling(window=20).sum()
    total_volume = df['volume'].rolling(window=20).sum()
    df['volume_weighted_gap'] = total_gap_volume_product / total_volume
    
    # Incorporate Price Momentum
    short_ema_window = 10
    df['ema_close'] = df['close'].ewm(span=short_ema_window, adjust=False).mean()
    df['change_in_ema'] = df['ema_close'] - df['ema_close'].shift(1)
    
    # Introduce ATR Component
    df['true_range'] = df[['high', 'low', 'close']].apply(
        lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift(1)), abs(x['low'] - x['close'].shift(1))), axis=1
    )
    atr_window = 14
    df['atr'] = df['true_range'].rolling(window=atr_window).mean()
    df['adjusted_momentum'] = df['change_in_ema'] / df['atr']
    
    # Combine Factors and Adjust for Volume
    volume_ma_window = 20
    df['volume_ma'] = df['volume'].rolling(window=volume_ma_window).mean()
    df['volume_adjusted_momentum'] = df['adjusted_momentum'] * df['volume_ma']
    
    # Price Reversal Sensitivity
    df['high_low_spread'] = df['high'] - df['low']
    df['weighted_high_low_spread'] = df['high_low_spread'] * df['volume']
    
    # Final Alpha Factor
    df['alpha_factor'] = (df['volume_weighted_gap'] + df['volume_adjusted_momentum']) - df['weighted_high_low_spread']
    
    return df['alpha_factor']
