import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics(df):
    # Intraday High-to-Low Range
    df['daily_range'] = df['high'] - df['low']
    
    # Open to Close Momentum
    df['open_to_close_return'] = (df['close'] - df['open']) / df['open']
    df['open_to_close_ema'] = df['open_to_close_return'].ewm(span=5, adjust=False).mean()
    
    # Volume-Weighted Open-to-Close Return
    df['volume_weighted_return'] = df['open_to_close_return'] * df['volume']
    df['volume_weighted_ema'] = df['volume_weighted_return'].ewm(span=5, adjust=False).mean()
    
    # Price-Volume Trend Indicator
    df['price_change'] = df['close'] - df['close'].shift(1)
    df['price_volume_trend'] = df['price_change'] * df['volume']
    df['pvt_30_day_sum'] = df['price_volume_trend'].rolling(window=30).sum()
    
    # Volume-Adjusted Intraday Movement
    df['intraday_movement'] = df['close'] - df['open']
    df['20_day_sma_volume'] = df['volume'].rolling(window=20).mean()
    df['volume_adjusted_intraday'] = df['intraday_movement'] / df['20_day_sma_volume']
    
    # Dynamic Volatility Measure
    df['20_day_volatility'] = df['close'].rolling(window=20).std()
    
    # Combined Momentum and Volatility Factor
    market_sentiment = df['20_day_volatility']  # Simplified market sentiment based on volatility
    df['combined_factor'] = (df['daily_range'] * (1 / market_sentiment) +
                             df['open_to_close_ema'] * (1 / market_sentiment) +
                             df['volume_weighted_ema'] * (1 / market_sentiment)) / 3
    
    # Adaptive Exponential Smoothing for Combined Momentum and Volatility Factor
    alpha = 1 / (1 + np.exp(-market_sentiment))
    df['combined_factor_smoothed'] = df['combined_factor'].ewm(alpha=alpha, adjust=False).mean()
    
    # Volume-Sensitive Momentum Factor
    df['volume_sensitive_momentum'] = (df['pvt_30_day_sum'] * (1 / market_sentiment) +
                                       df['volume_weighted_ema'] * (1 / market_sentiment) +
                                       df['volume_adjusted_intraday'] * (1 / market_sentiment)) / 3
    
    # Adaptive Exponential Smoothing for Volume-Sensitive Momentum Factor
    alpha = 1 / (1 + np.exp(-market_sentiment))
    df['volume_sensitive_momentum_smoothed'] = df['volume_sensitive_momentum'].ewm(alpha=alpha, adjust=False).mean()
    
    # Final Alpha Factor
    df['final_alpha'] = (df['combined_factor_smoothed'] * (1 / market_sentiment) +
                         df['volume_sensitive_momentum_smoothed'] * (1 / market_sentiment)) / 2
    
    # Adaptive Exponential Smoothing for Final Alpha Factor
    alpha = 1 / (1 + np.exp(-market_sentiment))
    df['final_alpha_smoothed'] = df['final_alpha'].ewm(alpha=alpha, adjust=False).mean()
    
    return df['final_alpha_smoothed']

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# final_alpha = heuristics(df)
# print(final_alpha)
