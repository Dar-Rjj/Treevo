import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate High-Low Spread
    high_low_spread = df['high'] - df['low']
    
    # Calculate Daily Volume Trend
    volume_10_day_ma = df['volume'].rolling(window=10).mean()
    volume_trend = df['volume'] - volume_10_day_ma
    volume_trend_sign = np.where(volume_trend > 0, 1, -1)
    
    # Calculate Price Trend
    close_ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    price_trend = df['close'] - close_ema_10
    price_trend_sign = np.where(price_trend > 0, 1, -1)
    
    # Calculate Volatility
    volatility = df['close'].rolling(window=10).std()
    volatility_level = np.where(volatility > volatility.mean(), 'high', 'low')
    
    # Adjust High-Low Spread based on Volume Trend
    adjusted_spread = high_low_spread * (1.5 if volume_trend_sign == 1 else 0.5)
    
    # Incorporate Price Trend
    adjusted_spread = adjusted_spread * (1.2 if price_trend_sign == 1 else 0.8)
    
    # Consider Volatility
    combined_factor = adjusted_spread
    combined_factor = np.where(volatility_level == 'high', combined_factor * 1.3, combined_factor * 0.7)
    
    return pd.Series(combined_factor, index=df.index, name='alpha_factor')

# Example usage:
# df = pd.read_csv('market_data.csv', parse_dates=['date'], index_col='date')
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
