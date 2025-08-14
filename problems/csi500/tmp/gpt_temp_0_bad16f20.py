import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the difference between high and low prices
    price_range = df['high'] - df['low']
    
    # Calculate the volume-weighted average of (high - close) and (close - low)
    vol_weighted_diff = (df['volume'] * ((df['high'] - df['close']) + (df['close'] - df['low'])) / 2)
    
    # Calculate a smoothed version of the volume-weighted difference with a longer window
    smooth_vol_weighted_diff = vol_weighted_diff.rolling(window=10).mean().fillna(0)
    
    # Calculate the amount-weighted average of (high - open) and (open - low)
    amount_weighted_diff = (df['amount'] * ((df['high'] - df['open']) + (df['open'] - df['low'])) / 2)
    
    # Calculate a smoothed version of the amount-weighted difference with a longer window
    smooth_amount_weighted_diff = amount_weighted_diff.rolling(window=10).mean().fillna(0)
    
    # Calculate the typical price
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate the volume-weighted typical price with a longer window
    vol_weighted_typical_price = (df['volume'] * typical_price).rolling(window=10).mean().fillna(0)
    
    # Calculate the amount-weighted typical price with a longer window
    amount_weighted_typical_price = (df['amount'] * typical_price).rolling(window=10).mean().fillna(0)
    
    # Additional price/volume ratios
    close_to_open_ratio = df['close'] / df['open']
    close_to_high_ratio = df['close'] / df['high']
    close_to_low_ratio = df['close'] / df['low']
    volume_to_amount_ratio = df['volume'] / (df['amount'] + 1e-7)
    
    # Momentum factor: 5-day return
    momentum_5d = (df['close'] / df['close'].shift(5)).fillna(1)
    
    # Momentum factor: 20-day return
    momentum_20d = (df['close'] / df['close'].shift(20)).fillna(1)
    
    # Volatility factor: 20-day standard deviation of returns
    daily_returns = df['close'].pct_change().fillna(0)
    volatility_20d = daily_returns.rolling(window=20).std().fillna(0)
    
    # Trend factor: 50-day moving average
    trend_50d = df['close'].rolling(window=50).mean().fillna(df['close'])
    
    # Seasonality factor: day of the week
    day_of_week = df.index.dayofweek + 1  # Monday=1, Sunday=7
    
    # Final alpha factor: combine the smoothed volume-weighted and amount-weighted differences,
    # the volume-weighted and amount-weighted typical prices, additional price/volume ratios,
    # and incorporate momentum, volatility, trend, and seasonality factors
    alpha_factor = (smooth_vol_weighted_diff + smooth_amount_weighted_diff + 
                    vol_weighted_typical_price + amount_weighted_typical_price +
                    close_to_open_ratio + close_to_high_ratio + close_to_low_ratio + 
                    volume_to_amount_ratio + 
                    momentum_5d + momentum_20d + 
                    volatility_20d + trend_50d + day_of_week) / (price_range + 1e-7)
    
    return alpha_factor
