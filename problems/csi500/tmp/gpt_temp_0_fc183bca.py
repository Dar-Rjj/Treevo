import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the difference between high and low prices
    price_range = df['high'] - df['low']
    
    # Calculate the volume-weighted average of (high - close) and (close - low)
    vol_weighted_diff = (df['volume'] * ((df['high'] - df['close']) + (df['close'] - df['low'])) / 2)
    
    # Calculate a smoothed version of the volume-weighted difference with a short window
    smooth_vol_weighted_diff = vol_weighted_diff.rolling(window=3).mean().fillna(0)
    
    # Calculate the amount-weighted average of (high - open) and (open - low)
    amount_weighted_diff = (df['amount'] * ((df['high'] - df['open']) + (df['open'] - df['low'])) / 2)
    
    # Calculate a smoothed version of the amount-weighted difference with a short window
    smooth_amount_weighted_diff = amount_weighted_diff.rolling(window=3).mean().fillna(0)
    
    # Calculate the typical price
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate the volume-weighted typical price
    vol_weighted_typical_price = (df['volume'] * typical_price).rolling(window=3).mean().fillna(0)
    
    # Calculate the amount-weighted typical price
    amount_weighted_typical_price = (df['amount'] * typical_price).rolling(window=3).mean().fillna(0)
    
    # Additional price/volume ratios
    close_to_open_ratio = df['close'] / df['open']
    
    # Calculate RSI over a short term
    rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    
    # Calculate momentum over a longer term
    momentum = df['close'].pct_change(periods=12).fillna(0)
    
    # Final alpha factor: combine the smoothed volume-weighted and amount-weighted differences,
    # the volume-weighted and amount-weighted typical prices, the close-to-open ratio, RSI, and momentum
    alpha_factor = (smooth_vol_weighted_diff + smooth_amount_weighted_diff + 
                    vol_weighted_typical_price + amount_weighted_typical_price +
                    close_to_open_ratio + rsi + momentum) / (price_range + 1e-7)
    
    return alpha_factor
