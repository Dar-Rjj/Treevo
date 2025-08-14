import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Adjust Intraday Range by Volume
    volume_ema = df['volume'].ewm(span=10).mean()
    adjusted_volume = df['volume'] / volume_ema
    intraday_range_adjusted = intraday_range * adjusted_volume
    
    # Further Adjustment by Close Price Volatility
    close_returns = df['close'].pct_change()
    close_volatility = close_returns.rolling(window=20).std()
    intraday_range_further_adjusted = intraday_range_adjusted / close_volatility
    
    # Calculate True Range
    true_range = np.maximum.reduce([df['high'] - df['low'], 
                                    abs(df['high'] - df['close'].shift(1)), 
                                    abs(df['low'] - df['close'].shift(1))])
    
    # Adjust Intraday Range by True Range Volatility
    true_range_volatility = true_range.rolling(window=20).std()
    intraday_range_final_adjusted = intraday_range_further_adjusted / true_range_volatility
    
    # Calculate Daily High-Low Difference
    high_low_diff = df['high'] - df['low']
    
    # Compute EMA of High-Low Difference
    high_low_ema = high_low_diff.ewm(span=10).mean()
    
    # Compute High-Low Momentum
    high_low_momentum = high_low_diff - high_low_ema.shift(1)
    
    # Calculate Price Momentum
    price_ma_10 = df['close'].rolling(window=10).mean()
    price_ma_20 = df['close'].rolling(window=20).mean()
    price_momentum = price_ma_10 - price_ma_20
    
    # Calculate Volume Spike
    volume_median_7 = df['volume'].rolling(window=7).median()
    volume_ratio = df['volume'] / volume_median_7
    
    # Enhance Price-Velocity Factors
    price_roc_5 = df['close'].pct_change(periods=5)
    price_roc_10 = df['close'].pct_change(periods=10)
    volume_roc_3 = df['volume'].pct_change(periods=3)
    volume_roc_7 = df['volume'].pct_change(periods=7)
    
    # Incorporate Open-Price Momentum
    open_ma_10 = df['open'].rolling(window=10).mean()
    open_ma_20 = df['open'].rolling(window=20).mean()
    open_price_momentum = open_ma_10 - open_ma_20
    
    # Synthesize Intraday, High-Low, and Price-Volume Momentum
    factor = (intraday_range_final_adjusted * volume_ratio +
              high_low_momentum * volume_ratio +
              price_momentum * volume_ratio +
              (price_roc_10 - price_roc_5) * volume_ratio +
              (volume_roc_7 - volume_roc_3) * volume_ratio +
              open_price_momentum * volume_ratio)
    
    return factor
