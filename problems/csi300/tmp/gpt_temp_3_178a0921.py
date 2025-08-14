import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the momentum factor using a dynamic lookback period (10-30 days) based on volatility
    volatility = np.log(df['close']).diff().rolling(window=40).std()
    lookback_period_momentum = 10 + (volatility * 20).astype(int)
    momentum = np.log(df['close']).pct_change(periods=lookback_period_momentum)
    
    # Calculate the liquidity factor using a dynamic lookback period (10-40 days) based on volume
    lookback_period_liquidity = 10 + (df['volume'].rolling(window=40).std() * 30).astype(int)
    liquidity = df['volume'].rolling(window=lookback_period_liquidity).mean()
    
    # Calculate the market sentiment using the average money flow ratio over a dynamic lookback period (5-15 days)
    money_flow_ratio = df['amount'] / df['volume']
    lookback_period_sentiment = 5 + (money_flow_ratio.rolling(window=5).std() * 10).astype(int)
    market_sentiment = money_flow_ratio.rolling(window=lookback_period_sentiment).mean()
    
    # Calculate the volatility factor using the standard deviation of the log returns over a dynamic lookback period (20-50 days)
    lookback_period_volatility = 20 + (volatility * 30).astype(int)
    volatility = np.log(df['close']).diff().rolling(window=lookback_period_volatility).std()
    
    # Calculate the intraday dynamics factor using the ratio of high to low prices over the last 5 days
    intraday_dynamics = (df['high'] / df['low']).rolling(window=5).mean()
    
    # Dynamic weighting based on the standard deviation of each factor
    std_momentum = momentum.std()
    std_liquidity = liquidity.std()
    std_market_sentiment = market_sentiment.std()
    std_volatility = volatility.std()
    std_intraday_dynamics = intraday_dynamics.std()
    
    total_std = std_momentum + std_liquidity + std_market_sentiment + std_volatility + std_intraday_dynamics
    weight_momentum = std_momentum / total_std
    weight_liquidity = std_liquidity / total_std
    weight_market_sentiment = std_market_sentiment / total_std
    weight_volatility = std_volatility / total_std
    weight_intraday_dynamics = std_intraday_dynamics / total_std
    
    # Create a combined factor by multiplying the weighted factors
    # The idea is that stocks with positive momentum, high liquidity, strong market sentiment, lower volatility, and positive intraday dynamics may outperform
    factor = (momentum + abs(momentum.min())) * weight_momentum * (liquidity / liquidity.mean()) * weight_liquidity * (market_sentiment / market_sentiment.mean()) * weight_market_sentiment * (intraday_dynamics / intraday_dynamics.mean()) * weight_intraday_dynamics / ((volatility + 1e-6) * weight_volatility)
    
    return factor
