import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close to High and Low Ratios
    close_high_ratio = df['close'] / df['high']
    close_low_ratio = df['close'] / df['low']
    
    # Combine Close to High and Low Ratios
    combined_ratio = 0.7 * close_high_ratio + 0.3 * close_low_ratio
    
    # Calculate Intraday Price Movements
    high_low_diff = df['high'] - df['low']
    high_low_pct = (df['high'] - df['low']) / df['open']
    
    # Smooth Intraday Percentage Movement
    smoothed_pct_move = high_low_pct.ewm(span=5).mean()
    
    # Calculate Volume-Adjusted Momentum
    price_change = df['close'].diff(1)
    volume_adj_momentum = price_change * np.sqrt(df['volume'])
    
    # Calculate Volume Change
    vol_change = df['volume'].diff(1)
    vol_change_pct = vol_change / df['volume'].shift(1)
    
    # Apply Volume Shock Filter
    vol_shock_threshold = vol_change.abs().quantile(0.85)
    filtered_df = df[vol_change.abs() < vol_shock_threshold]
    
    # Calculate Intraday Return
    intraday_return = (df['close'] - df['open']) / df['open']
    
    # Compute EMA of Intraday Return
    ema_intraday_return = intraday_return.ewm(span=5).mean()
    
    # Incorporate Volatility Component
    intraday_range = df['high'] - df['low']
    volatility = intraday_range.rolling(window=20).std()
    volatility_adjusted_momentum = volume_adj_momentum / volatility
    
    # Calculate Price Momentum
    log_returns = np.log(df['close'] / df['close'].shift(1))
    price_momentum = log_returns.rolling(window=20).sum()
    
    # Adjust by Volume Volatility
    daily_vol_change = df['volume'] - df['volume'].shift(1)
    volume_volatility = daily_vol_change.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    final_adjustment = volume_adj_momentum / volume_volatility
    
    # Calculate Daily Price Change
    daily_price_change = df['close'] - df['close'].shift(1)
    
    # Calculate Volume-Weighted Momentum
    volume_weighted_momentum = daily_price_change * np.log(df['volume'])
    
    # Compute 20-Day Rolling Sum of Volume-Weighted Momentum
    rolling_volume_weighted_momentum = volume_weighted_momentum.rolling(window=20).sum()
    
    # Adjust by Intraday High-Low Momentum
    adjusted_rolling_momentum = rolling_volume_weighted_momentum * final_adjustment
    
    # Incorporate Price Trend
    price_change_trend = df['close'] - df['open'].shift(1)
    long_term_price_trend = df['close'].rolling(window=200).mean()
    trend_adjusted_momentum = adjusted_rolling_momentum * price_change_trend * long_term_price_trend
    
    # Compute Dynamic Momentum Oscillator
    dynamic_momentum_oscillator = trend_adjusted_momentum - trend_adjusted_momentum.rolling(window=50).mean()
    
    return dynamic_momentum_oscillator
