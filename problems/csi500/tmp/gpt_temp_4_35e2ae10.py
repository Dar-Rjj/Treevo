import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate the difference between closing price and opening price
    daily_price_diff = df['close'] - df['open']
    
    # 5-day moving average of the daily price difference
    ma_5_daily_diff = daily_price_diff.rolling(window=5).mean()
    
    # 20-day standard deviation of the 5-day moving average of the daily price difference
    std_20_daily_diff = ma_5_daily_diff.rolling(window=20).std()
    
    # Volatility-adjusted trend strength
    volatility_adjusted_trend = ma_5_daily_diff / std_20_daily_diff
    
    # Count the number of days in the past month where the closing price was above the opening price
    up_days = (df['close'] > df['open']).rolling(window=20).sum()
    
    # Ratio of up-days to total days
    up_day_ratio = up_days / 20
    
    # Calculate High-Low Spread
    high_low_spread = df['high'] - df['low']
    
    # Volume-Average Price
    volume_average_price = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Volume-Averaged High-Low Spread Ratio
    volume_averaged_high_low_ratio = high_low_spread / volume_average_price
    
    # 14-day Exponential Moving Average (EMA) of the closing prices
    ema_14_close = df['close'].ewm(span=14, adjust=False).mean()
    
    # Difference between the current closing price and the 14-day EMA
    ema_diff = df['close'] - ema_14_close
    
    # Highest high and lowest low over the last 20 days
    highest_high_20 = df['high'].rolling(window=20).max()
    lowest_low_20 = df['low'].rolling(window=20).min()
    
    # Range percentage
    range_percentage = (highest_high_20 - lowest_low_20) / lowest_low_20
    
    # Sum of volume on days when the closing price is higher than the opening price, divided by the total volume over the same period
    volume_up_days = (df['close'] > df['open']) * df['volume']
    buying_pressure_ratio = volume_up_days.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # Rate of change of the amount traded over the past 7 days
    amount_roc = df['amount'].pct_change(periods=7)
    
    # Price gap (difference between today's open and yesterday's close)
    price_gap = df['open'] - df['close'].shift(1)
    
    # Categorize the gaps into up, down, or no gap
    gap_category = pd.cut(price_gap, bins=[-float('inf'), 0, float('inf')], labels=['down', 'no_gap', 'up'])
    
    # Day's high and the 20-day moving average of the closing price
    ma_20_close = df['close'].rolling(window=20).mean()
    high_above_ma_20 = (df['high'] > ma_20_close).astype(int)
    
    # Combine all factors into a single alpha factor
    alpha_factor = (
        0.2 * volatility_adjusted_trend +
        0.1 * up_day_ratio +
        0.1 * volume_averaged_high_low_ratio +
        0.1 * ema_diff +
        0.1 * range_percentage +
        0.1 * buying_pressure_ratio +
        0.1 * amount_roc +
        0.1 * high_above_ma_20
    )
    
    return alpha_factor
