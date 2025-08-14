import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday High-Low Momentum
    high_low_diff = df['high'] - df['low']
    intraday_momentum_ema = high_low_diff.ewm(span=5, adjust=False).mean()
    
    # Adjust by Volume Volatility
    daily_volume_change = df['volume'] - df['volume'].shift(1)
    volume_volatility = daily_volume_change.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    final_adjustment = intraday_momentum_ema / volume_volatility
    
    # Calculate Daily Price Change
    daily_price_change = df['close'] - df['close'].shift(1)
    
    # Calculate Volume-Weighted Momentum
    volume_weighted_momentum = daily_price_change * np.log(df['volume'])
    
    # Compute 20-Day Rolling Sum of Volume-Weighted Momentum
    rolling_sum_vwm = volume_weighted_momentum.rolling(window=20).sum()
    
    # Adjust by Intraday High-Low Momentum
    intraday_adjusted_vwm = rolling_sum_vwm * final_adjustment
    
    # Incorporate Price Trend
    price_trend = df['close'] - df['open'].shift(1)
    trend_adjusted_vwm = intraday_adjusted_vwm * price_trend
    
    # Compute Dynamic Momentum Oscillator
    dynamic_momentum_oscillator = trend_adjusted_vwm - trend_adjusted_vwm.rolling(window=50).mean()
    
    # Analyze Closing Prices
    close_diff = df['close'] - df['close'].shift(20)
    ma_cross_above = (df['close'] > df['close'].rolling(window=20).mean()) & (df['close'].shift(1) < df['close'].rolling(window=20).mean().shift(1))
    ma_cross_below = (df['close'] < df['close'].rolling(window=20).mean()) & (df['close'].shift(1) > df['close'].rolling(window=20).mean().shift(1))
    
    # Examine Volume Trends
    volume_diff = df['volume'] - df['volume'].shift(20)
    significant_volume_increase = (df['volume'] > df['volume'].shift(1) * 1.5)
    
    # Consider Open-High-Low-Close (OHLC) Relationships
    ohlc_range = df['high'] - df['low']
    doji_pattern = (df['close'] - df['open']).abs() / ohlc_range < 0.05
    hammer_pattern = (df['close'] - df['low']) / ohlc_range > 0.6 & (df['open'] - df['low']) / ohlc_range > 0.6
    engulfing_bullish = (df['close'] > df['open']) & (df['open'] < df['open'].shift(1)) & (df['close'] > df['close'].shift(1))
    engulfing_bearish = (df['close'] < df['open']) & (df['open'] > df['open'].shift(1)) & (df['close'] < df['close'].shift(1))
    
    # Incorporate Transaction Amounts
    amount_diff = df['amount'] - df['amount'].shift(20)
    large_transaction = df['amount'] > df['amount'].rolling(window=20).mean() * 2
    
    # Combine all factors into a single alpha factor
    alpha_factor = dynamic_momentum_oscillator + close_diff + volume_diff + \
                   (ma_cross_above.astype(int) - ma_cross_below.astype(int)) + \
                   (significant_volume_increase.astype(int)) + \
                   (doji_pattern.astype(int) + hammer_pattern.astype(int) + engulfing_bullish.astype(int) + engulfing_bearish.astype(int)) + \
                   (large_transaction.astype(int))
    
    return alpha_factor
