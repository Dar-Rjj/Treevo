import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the difference between high and low prices
    price_range = df['high'] - df['low']
    
    # Calculate the volume-weighted average of (high - close) and (close - low)
    vol_weighted_diff = (df['volume'] * ((df['high'] - df['close']) + (df['close'] - df['low'])) / 2)
    
    # Calculate a smoothed version of the volume-weighted difference with a shorter window
    smooth_vol_weighted_diff = vol_weighted_diff.rolling(window=1).mean().fillna(0)
    
    # Calculate the amount-weighted average of (high - open) and (open - low)
    amount_weighted_diff = (df['amount'] * ((df['high'] - df['open']) + (df['open'] - df['low'])) / 2)
    
    # Calculate a smoothed version of the amount-weighted difference with a shorter window
    smooth_amount_weighted_diff = amount_weighted_diff.rolling(window=1).mean().fillna(0)
    
    # Calculate the typical price
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate the volume-weighted typical price
    vol_weighted_typical_price = (df['volume'] * typical_price).rolling(window=1).mean().fillna(0)
    
    # Calculate the amount-weighted typical price
    amount_weighted_typical_price = (df['amount'] * typical_price).rolling(window=1).mean().fillna(0)
    
    # Additional price/volume ratios
    close_to_open_ratio = df['close'] / df['open']
    volume_to_amount_ratio = df['volume'] / df['amount']
    close_to_volume_ratio = df['close'] / df['volume']
    
    # Momentum factor: 5-day return
