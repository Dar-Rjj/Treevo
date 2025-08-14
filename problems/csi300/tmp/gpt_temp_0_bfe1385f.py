import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the 10-day and 30-day moving average of the closing price
    ma_close_10 = df['close'].rolling(window=10).mean()
    ma_close_30 = df['close'].rolling(window=30).mean()
    
    # Calculate the 10-day and 30-day moving average of the volume
    ma_volume_10 = df['volume'].rolling(window=10).mean()
    ma_volume_30 = df['volume'].rolling(window=30).mean()
    
    # Calculate the exponential moving average of the amount with a span of 5 days and 20 days
    ema_amount_5 = df['amount'].ewm(span=5, adjust=False).mean()
    ema_amount_20 = df['amount'].ewm(span=20, adjust=False).mean()
    
    # Calculate the ratio of today's close price to the 10-day and 30-day moving average of the closing price
    close_to_ma_close_ratio_10 = df['close'] / ma_close_10
    close_to_ma_close_ratio_30 = df['close'] / ma_close_30
    
    # Calculate the ratio of today's volume to the 10-day and 30-day moving average of the volume
    volume_to_ma_volume_ratio_10 = df['volume'] / ma_volume_10
    volume_to_ma_volume_ratio_30 = df['volume'] / ma_volume_30
    
    # Calculate the ratio of today's amount to the EMA of the amount
    amount_to_ema_amount_ratio_5 = df['amount'] / ema_amount_5
    amount_to_ema_amount_ratio_20 = df['amount'] / ema_amount_20
    
    # Calculate the 10-day and 30-day standard deviation of the closing price for volatility
    std_close_10 = df['close'].rolling(window=10).std()
    std_close_30 = df['close'].rolling(window=30).std()
    
    # Calculate the 10-day and 30-day ATR (Average True Range) for trend strength
    tr = df[['high', 'low', 'close']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift(1)), abs(x['low'] - x['close'].shift(1))), axis=1)
    atr_10 = tr.rolling(window=10).mean()
    atr_30 = tr.rolling(window=30).mean()
    
    # Generate the alpha factor by combining the ratios, EMA of the amount, and ATR
    alpha_factor = (
        (close_to_ma_close_ratio_10 - 1) * volume_to_ma_volume_ratio_10 +
        (close_to_ma_close_ratio_30 - 1) * volume_to_ma_volume_ratio_30 +
        (amount_to_ema_amount_ratio_5 - 1) + 
        (amount_to_ema_amount_ratio_20 - 1) +
        (std_close_10 / std_close_30) +
        (atr_10 / atr_30)
    ) / 6

    return alpha_factor
