import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # VWAP Calculation
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    # VWAP Differences
    vwap_diff_open = vwap - df['open']
    vwap_diff_close = vwap - df['close']
    
    # Raw Returns
    raw_returns = df['close'].pct_change()
    
    # 20-Day Sum of Upward and Downward Returns
    positive_sum_20 = raw_returns.rolling(window=20).apply(lambda x: x[x > 0].sum(), raw=False)
    negative_sum_20 = raw_returns.rolling(window=20).apply(lambda x: x[x < 0].abs().sum(), raw=False)
    
    # Relative Strength
    relative_strength = positive_sum_20 / negative_sum_20.replace(0, np.nan)
    
    # Smooth with EMA on Volume
    ema_volume = df['volume'].ewm(span=20, adjust=False).mean()
    smoothed_relative_strength = relative_strength * ema_volume
    
    # Momentum
    short_term_momentum = df['close'] - df['close'].shift(15)
    long_term_momentum = df['close'] - df['close'].shift(70)
    
    # Volatility
    short_term_volatility = df['close'].pct_change().rolling(window=10).std()
    long_term_volatility = df['close'].pct_change().rolling(window=60).std()
    
    # Combine Momentum and Volatility
    short_term_combined = short_term_momentum + short_term_volatility
    long_term_combined = long_term_momentum + long_term_volatility
    
    # Adjust Relative Strength with Price Trend
    price_trend = df['close'] / df['close'].rolling(window=21).mean()
    adjusted_relative_strength = smoothed_relative_strength * price_trend
    
    # Integrate VWAP Differences
    vwap_diff_open_adjusted = vwap_diff_open * adjusted_relative_strength
    vwap_diff_close_adjusted = vwap_diff_close * adjusted_relative_strength
    
    # Final Alpha Factor
    final_alpha_factor = (
        -long_term_combined + 
        short_term_combined + 
        vwap_diff_open_adjusted + 
        vwap_diff_close_adjusted
    )
    
    # Enhance Alpha Factor
    high_low_ratio = df['high'].rolling(window=10).max() / df['low'].rolling(window=10).min()
    enhanced_alpha_factor = final_alpha_factor * adjusted_relative_strength + high_low_ratio
    
    return enhanced_alpha_factor
