import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    import pandas as pd
    import numpy as np

    # Momentum calculation - 50 period return
    momentum = df['close'].pct_change(50)
    
    # Liquidity - Calculate the average volume over a 30 day period
    liquidity = df['volume'].rolling(window=30).mean()
    
    # Volatility - Calculate the standard deviation of daily returns over a 20 day period
    daily_returns = df['close'].pct_change()
    volatility = daily_returns.rolling(window=20).std()
    
    # True Range (TR) for enhanced volatility measure
    prev_close = df['close'].shift(1)
    tr = (df['high'] - df['low']).abs()
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (prev_close - df['low']).abs()
    avg_true_range = (tr + tr2 + tr3) / 3
    
    # Money Flow Index (MFI) with a 14-day period for market sentiment
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_money_flow = raw_money_flow.where(typical_price > prev_close, 0).rolling(window=14).sum()
    negative_money_flow = raw_money_flow.where(typical_price < prev_close, 0).rolling(window=14).sum()
    mfi = 100 - (100 / (1 + positive_money_flow / (negative_money_flow + 1e-7)))
    
    # Incorporate macroeconomic indicators (assuming they are available in the DataFrame as 'macro_indicator')
    macro_adjustment = df['macro_indicator'] / df['macro_indicator'].rolling(window=20).mean()
    
    # Sentiment analysis (assuming sentiment scores are available in the DataFrame as 'sentiment_score')
    sentiment_factor = df['sentiment_score'].rolling(window=10).mean()
    
    # Machine learning dynamic factor adjustment (assuming a model is available to predict a factor)
    # For simplicity, we'll use a placeholder here
    ml_factor = np.random.randn(len(df))
    
    # Composite alpha factor
    alpha_factor = (momentum * liquidity / (volatility + 1e-7)) * (mfi / 100) * (avg_true_range / (df['close'] + 1e-7)) * macro_adjustment * sentiment_factor * ml_factor
    return alpha_factor
