import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Momentum calculation - 50 period return
    momentum = df['close'].pct_change(50)
    
    # Adaptive window for liquidity - Calculate the average volume over a dynamic 30 day period
    liquidity = df['volume'].rolling(window=30, min_periods=1).mean()
    
    # Volatility - Calculate the standard deviation of daily log returns over a 20 day period
    daily_log_returns = np.log(df['close'] / df['close'].shift(1))
    volatility = daily_log_returns.rolling(window=20, min_periods=1).std()
    
    # True Range (TR) calculation for volatility
    prev_close = df['close'].shift(1)
    tr = df[['high', 'low']].apply(lambda x: (x[0] - x[1]), axis=1)
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (prev_close - df['low']).abs()
    avg_true_range = (tr + tr2 + tr3) / 3
    
    # Money Flow Index (MFI) with a 14-day period
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_money_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14, min_periods=1).sum()
    negative_money_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14, min_periods=1).sum()
    mfi = 100 - (100 / (1 + positive_money_flow / (negative_money_flow + 1e-7)))
    
    # Market Sentiment - Calculate the ratio of high to low prices
    sentiment = df['high'] / df['low']
    
    # Seasonality - Monthly seasonality factor
    monthly_seasonality = df.index.month.map({1: 1.05, 2: 1.04, 3: 1.03, 4: 1.02, 5: 1.01, 6: 1.0, 
                                              7: 0.99, 8: 0.98, 9: 0.97, 10: 0.96, 11: 0.95, 12: 0.94})
    
    # Dynamic price-volume interaction
    price_volume_interaction = (df['close'] * df['volume']).rolling(window=10, min_periods=1).mean()
    
    # Composite alpha factor
    alpha_factor = (momentum * liquidity / (volatility + 1e-7)) * (mfi / 100) * (sentiment - 1) * monthly_seasonality * (price_volume_interaction / df['volume'])
    return alpha_factor
