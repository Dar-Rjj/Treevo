import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame, market_cap: pd.Series) -> pd.Series:
    # Seasonality adjustment using month and day of the week
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    monthly_seasonality = df.groupby('month')['close'].transform(lambda x: (x - x.mean()) / x.std())
    daily_seasonality = df.groupby('day_of_week')['close'].transform(lambda x: (x - x.mean()) / x.std())

    # Log returns for a more normally distributed variable
    log_returns = np.log(df['close']).diff()

    # Liquidity - Calculate the average volume over a 60 day period
    liquidity = df['volume'].rolling(window=60).mean()

    # Volatility - Calculate the standard deviation of daily log returns over a 50 day period
    volatility = log_returns.rolling(window=50).std()

    # True Range (TR) calculation for volatility
    prev_close = df['close'].shift(1)
    tr = (df['high'] - df['low']).abs()
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (prev_close - df['low']).abs()
    avg_true_range = (tr + tr2 + tr3) / 3

    # Money Flow Index (MFI) with a 28-day period
    money_flow = df['amount'] * ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-7)
    positive_money_flow = money_flow.where(money_flow > 0, 0).rolling(window=28).sum()
    negative_money_flow = money_flow.abs().where(money_flow < 0, 0).rolling(window=28).sum()
    mfi = 100 - (100 / (1 + positive_money_flow / (negative_money_flow + 1e-7)))

    # Additional factor: Moving Average Convergence Divergence (MACD)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_diff = macd - signal

    # Incorporate market cap as a factor
    market_cap_factor = (market_cap / market_cap.shift(1)).fillna(1) - 1

    # Composite alpha factor
    alpha_factor = (log_returns.rolling(window=100).mean() * liquidity / (volatility + 1e-7)) * (mfi / 100) * (macd_diff / (avg_true_range + 1e-7)) * (1 + market_cap_factor) * (monthly_seasonality + daily_seasonality)
    return alpha_factor
