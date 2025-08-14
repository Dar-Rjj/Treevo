import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the momentum factor using log returns over the last 10 days
    log_returns = np.log(df['close'] / df['close'].shift(1))
    momentum = log_returns.rolling(window=10).sum()
    
    # Calculate the sector-specific liquidity factor using the average volume over the last 20 days
    # Assuming 'sector' is a column in the DataFrame, otherwise, you need to provide the sector information
    sector_liquidity = df.groupby('sector')['volume'].transform(lambda x: x.rolling(window=20).mean())
    liquidity = df['volume'] / sector_liquidity
    
    # Calculate the market sentiment using the money flow ratio over the last 5 days
    money_flow_ratio = df['amount'] / df['volume']
    market_sentiment = money_flow_ratio.rolling(window=5).mean()
    
    # Refine market sentiment by incorporating high-low price range
    high_low_range = df['high'] - df['low']
    refined_market_sentiment = (market_sentiment / high_low_range).rolling(window=5).mean()
    
    # Create a combined factor by multiplying the momentum, liquidity, and refined market sentiment
    # The idea is that stocks with positive momentum, high liquidity, and strong market sentiment may outperform
    factor = (momentum + 1) * (liquidity / liquidity.mean()) * (refined_market_sentiment / refined_market_sentiment.mean())
    
    return factor
