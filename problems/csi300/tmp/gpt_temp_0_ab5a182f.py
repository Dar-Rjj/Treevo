import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday High-to-Low Range
    daily_range = (df['high'] - df['low']) / df['open']
    
    # Open to Close Momentum
    open_to_close_return = (df['close'] - df['open']) / df['open']
    otc_sma_5 = open_to_close_return.rolling(window=5).mean()
    
    # Volume Adjusted Intraday Movement
    intraday_movement = (df['close'] - df['open'])
    avg_volume_20 = df['volume'].rolling(window=20).mean()
    volume_adjusted_movement = intraday_movement / avg_volume_20
    
    # Price-Volume Trend Indicator
    daily_price_change = df['close'].diff(1)
    price_volume_trend = (daily_price_change * df['volume']).rolling(window=30).sum()
    
    # Combined Momentum and Volatility Factor
    combined_momentum_volatility = 0.5 * daily_range + 0.5 * otc_sma_5
    combined_momentum_volatility = combined_momentum_volatility.ewm(alpha=0.2).mean()
    
    # Volume-Sensitive Momentum Factor
    volume_sensitive_momentum = 0.5 * price_volume_trend + 0.5 * volume_adjusted_movement
    volume_sensitive_momentum = volume_sensitive_momentum.ewm(alpha=0.2).mean()
    
    # Relative Strength Indicator
    relative_strength = df['close'] / df['close_benchmark']
    rsi_sma_14 = relative_strength.rolling(window=14).mean()
    
    # Average True Range
    true_range = pd.concat([
        (df['high'] - df['low']),
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_sma_14 = true_range.rolling(window=14).mean()
    
    # Final Alpha Factor
    final_alpha_factor = 0.2 * combined_momentum_volatility + 0.2 * volume_sensitive_momentum + 0.2 * rsi_sma_14 + 0.2 * atr_sma_14
    final_alpha_factor = final_alpha_factor.ewm(alpha=0.2).mean()
    
    return final_alpha_factor
