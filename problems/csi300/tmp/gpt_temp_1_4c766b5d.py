import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Momentum calculation - 100 period return
    momentum = df['close'].pct_change(100)
    
    # Liquidity - Calculate the average volume over a 60 day period
    liquidity = df['volume'].rolling(window=60).mean()
    
    # Volatility - Calculate the standard deviation of daily returns over a 30 day period
    daily_returns = df['close'].pct_change()
    volatility = daily_returns.rolling(window=30).std()
    
    # True Range (TR) calculation for enhanced volatility
    prev_close = df['close'].shift(1)
    tr = df[['high', 'low']].apply(lambda x: (x[0] - x[1]), axis=1)
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (prev_close - df['low']).abs()
    true_range = tr.combine(tr2, max).combine(tr3, max)
    avg_true_range = true_range.rolling(window=14).mean()
    
    # Money Flow Index (MFI) with a 14-day period
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_money_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
    negative_money_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
    mfi = 100 - (100 / (1 + positive_money_flow / (negative_money_flow + 1e-7)))
    
    # Composite alpha factor
    alpha_factor = (momentum * liquidity / (volatility + 1e-7)) * (mfi / 100) * (avg_true_range + 1e-7)
    return alpha_factor
