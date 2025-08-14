import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday High-to-Low Range
    df['daily_range'] = (df['high'] - df['low']) / df['open']
    
    # Open to Close Momentum
    df['open_to_close_return'] = (df['close'] - df['open']) / df['open']
    df['open_to_close_ema5'] = df['open_to_close_return'].ewm(span=5, adjust=False).mean()
    
    # Volume-Weighted Open-to-Close Return
    df['volume_weighted_return'] = df['open_to_close_return'] * df['volume']
    df['volume_weighted_ema5'] = df['volume_weighted_return'].ewm(span=5, adjust=False).mean()
    
    # Price-Volume Trend Indicator
    df['price_change'] = df['close'] - df['close'].shift(1)
    df['price_volume_trend'] = df['price_change'] * df['volume']
    df['pvt_30d'] = df['price_volume_trend'].rolling(window=30).sum()
    
    # Volume-Adjusted Intraday Movement
    df['intraday_movement'] = df['close'] - df['open']
    df['vol_20_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_adjusted_intraday'] = df['intraday_movement'] / df['vol_20_sma']
    
    # Combined Momentum and Volatility Factor
    # Placeholder for dynamic weights
    liquidity_factor = 1.0  # Example: Replace with actual liquidity factor calculation
    leverage_factor = 1.0   # Example: Replace with actual leverage factor calculation
    rsi = 1.0               # Example: Replace with actual RSI calculation
    combined_momentum_volatility = (0.4 * liquidity_factor * df['daily_range'] +
                                    0.3 * leverage_factor * df['open_to_close_ema5'] +
                                    0.3 * rsi * df['volume_weighted_ema5'])
    combined_momentum_volatility_smoothed = combined_movement_volatility.ewm(alpha=0.2, adjust=False).mean()
    
    # Volume-Sensitive Momentum Factor
    macd = 1.0  # Example: Replace with actual MACD calculation
    volume_sensitive_momentum = (0.4 * liquidity_factor * df['pvt_30d'] +
                                 0.3 * leverage_factor * df['volume_weighted_ema5'] +
                                 0.3 * macd * df['volume_adjusted_intraday'])
    volume_sensitive_momentum_smoothed = volume_sensitive_momentum.ewm(alpha=0.2, adjust=False).mean()
    
    # Final Alpha Factor
    final_alpha = (0.6 * liquidity_factor * combined_momentum_volatility_smoothed +
                   0.4 * leverage_factor * volume_sensitive_momentum_smoothed)
    final_alpha_smoothed = final_alpha.ewm(alpha=0.2, adjust=False).mean()
    
    return final_alpha_smoothed

# Example usage:
# df = pd.DataFrame({
#     'open': [100, 101, 102, ...],
#     'high': [105, 106, 107, ...],
#     'low': [98, 99, 100, ...],
#     'close': [103, 104, 105, ...],
#     'amount': [...],
#     'volume': [...]
# }, index=pd.to_datetime([...]))
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
