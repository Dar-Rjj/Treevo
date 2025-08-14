import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the 5-day and 20-day exponential moving averages of the close price
    ema_5 = df['close'].ewm(span=5, adjust=False).mean()
    ema_20 = df['close'].ewm(span=20, adjust=False).mean()
    
    # Calculate a weighted sum of the open, high, low, and close prices with adaptive weights
    weighted_prices = (0.1 * df['open'] + 0.3 * df['high'] + 0.3 * df['low'] + 0.3 * df['close'])
    
    # Calculate the ratio of range (high - low) to the 5-day EMA of the close price
    volatility_ratio = (df['high'] - df['low']) / (ema_5 + 1e-7)
    
    # Calculate the 5-day realized volatility using the standard deviation of log returns
    log_returns = np.log(df['close'] / df['close'].shift(1))
    realized_volatility = log_returns.rolling(window=5).std()
    
    # Calculate the 5-day average trading amount and volume
    avg_amount = df['amount'].rolling(window=5).mean()
    avg_volume = df['volume'].rolling(window=5).mean()
    
    # Trend strength factor
    trend_strength = (ema_5 - ema_20) / (realized_volatility + 1e-7)
    
    # Liquidity factor
    liquidity = (avg_amount / avg_volume) * (volatility_ratio + 1e-7)
    
    # Volume trend factor
    volume_trend = (df['volume'] / avg_volume) - 1
    
    # Adaptive weights based on trend strength and realized volatility
    adaptive_weight_trend = (trend_strength > 0).astype(int) * 0.5 + (realized_volatility < 0.05).astype(int) * 0.5
    adaptive_weight_volatility = (realized_volatility < 0.05).astype(int)
    
    # Momentum factor
    momentum = (df['close'] - df['close'].shift(20)) / (df['close'].shift(20) + 1e-7)
    
    # Seasonality factor
    month = df.index.month
    seasonality_factor = (month >= 10) | (month <= 4)  # Example: Positive seasonality in Q4 and Q1
    
    # Macroeconomic indicators (example: use a simple dummy variable for illustration)
    macro_indicator = (df['close'] > df['close'].shift(1)).astype(int)  # Simplified macro indicator
    
    # Cross-asset correlation (example: use a simple dummy variable for illustration)
    cross_asset_correlation = (df['close'] > df['close'].shift(1)).astype(int)  # Simplified cross-asset correlation
    
    # Final alpha factor as a combination of the above metrics with adaptive weights
    alpha_factor = (adaptive_weight_trend * (weighted_prices - ema_5) / (volatility_ratio + 1e-7)) + \
                   (1 - adaptive_weight_trend) * (trend_strength + liquidity + volume_trend) + \
                   adaptive_weight_volatility * (realized_volatility + 1e-7) + \
                   0.1 * momentum + \
                   0.1 * seasonality_factor + \
                   0.1 * macro_indicator + \
                   0.1 * cross_asset_correlation
    
    return alpha_factor
