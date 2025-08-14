import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the momentum factor using the change in close price over the last 10 days
    momentum = df['close'].pct_change(periods=10)
    
    # Calculate the liquidity factor using the average volume over the last 20 days
    liquidity = df['volume'].rolling(window=20).mean()
    
    # Calculate the market sentiment using the average money flow ratio over the last 5 days
    money_flow_ratio = df['amount'] / df['volume']
    market_sentiment = money_flow_ratio.rolling(window=5).mean()
    
    # Calculate the volatility factor using the standard deviation of the close price over the last 30 days
    volatility = df['close'].rolling(window=30).std()
    
    # Incorporate a sector-specific factor, for example, relative strength to its sector
    # Assume 'sector_returns' is a column in the DataFrame representing the sector returns
    relative_strength = df['close'].pct_change(periods=10) - df['sector_returns'].pct_change(periods=10)
    
    # Incorporate a macroeconomic indicator, for example, the 10-year Treasury yield
    # Assume 'treasury_yield' is a column in the DataFrame representing the 10-year Treasury yield
    macro_indicator = df['treasury_yield']
    
    # Create a combined factor by integrating the momentum, liquidity, market sentiment, relative strength, and macroeconomic indicator
    # The idea is that stocks with positive momentum, high liquidity, strong market sentiment, strong relative strength, and favorable macro conditions may outperform
    factor = (momentum + 1) * (liquidity / liquidity.mean()) * (market_sentiment / market_sentiment.mean()) * (relative_strength + 1) * (macro_indicator / macro_indicator.mean())
    
    return factor
