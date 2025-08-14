import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the volatility as the standard deviation of daily returns over a 10-day window
    df['daily_return'] = df['close'].pct_change()
    df['volatility_10d'] = df['daily_return'].rolling(window=10).std()

    # Calculate the average true range (ATR) to capture the trading activity
    df['true_range'] = df[['high' - 'low', 'high' - 'close_shift', 'close_shift' - 'low']].max(axis=1)
    df['atr_14d'] = df['true_range'].rolling(window=14).mean()
    df['close_shift'] = df['close'].shift(1)

    # Calculate the RSI (Relative Strength Index) to measure market sentiment over a 14-day window
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Calculate the difference between closing price and the average of open and close prices
    avg_price_diff = df['close'] - (df['open'] + df['close']) / 2

    # Calculate the normalized amount by dividing it with the volume, adding a small constant to avoid division by zero
    normalized_amount = df['amount'] / (df['volume'] + 1e-7)

    # Combine the metrics to generate a composite factor
    composite_factor = (avg_price_diff * normalized_amount) / (df['volatility_10d'] + 1e-7) * df['atr_14d'] * rsi

    return composite_factor
