import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Momentum calculation - 100 period return for a longer-term trend
    momentum = df['close'].pct_change(100)
    
    # Liquidity - Calculate the average volume over a 60 day period to smooth out short-term fluctuations
    liquidity = df['volume'].rolling(window=60).mean()
    
    # Volatility - Calculate the standard deviation of daily returns over a 30 day period for a more stable measure
    daily_returns = df['close'].pct_change()
    volatility = daily_returns.rolling(window=30).std()
    
    # True Range (TR) calculation for volatility
    prev_close = df['close'].shift(1)
    tr = df[['high', 'low']].apply(lambda x: (x[0] - x[1]), axis=1)
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (prev_close - df['low']).abs()
    avg_true_range = (tr + tr2 + tr3).rolling(window=30).mean()  # Use rolling mean for TR
    
    # Money Flow Index (MFI) with a 20-day period
    money_flow = df['amount'] * ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-7)
    positive_money_flow = money_flow.where(money_flow > 0, 0).rolling(window=20).sum()
    negative_money_flow = money_flow.abs().where(money_flow < 0, 0).rolling(window=20).sum()
    mfi = 100 - (100 / (1 + positive_money_flow / (negative_money_flow + 1e-7)))
    
    # Seasonality - Calculate the monthly return and use it as a factor
    monthly_return = df['close'].resample('M').ffill().pct_change().reindex(df.index).fillna(0)
    
    # Sentiment - Assume we have a sentiment score for each date (column 'sentiment' in the DataFrame)
    sentiment = df['sentiment']
    
    # Machine Learning Predictions - Assume we have a machine learning prediction for each date (column 'ml_prediction' in the DataFrame)
    ml_prediction = df['ml_prediction']
    
    # Composite alpha factor
    alpha_factor = (momentum * liquidity / (volatility + 1e-7)) * (mfi / 100) * (avg_true_range / df['close']) * (monthly_return + 1) * (sentiment + 1) * ml_prediction
    return alpha_factor
