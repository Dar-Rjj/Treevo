import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame, sector: str) -> pd.Series:
    # Define dynamic lookback periods based on sector
    momentum_lookbacks = (10, 20, 30) if sector == 'Tech' else (15, 25, 35)
    liquidity_lookbacks = (10, 20, 30, 40) if sector == 'Tech' else (15, 25, 35, 45)
    market_sentiment_lookbacks = (5, 10, 15) if sector == 'Tech' else (7, 12, 17)
    volatility_lookbacks = (20, 40, 60) if sector == 'Tech' else (30, 50, 70)
    intraday_dynamics_lookbacks = (5, 10) if sector == 'Tech' else (7, 12)

    # Calculate the momentum factor using the log return over a dynamic lookback period
    momentum_factors = [np.log(df['close']).diff(periods=lookback) for lookback in momentum_lookbacks]
    momentum = sum(momentum_factors) / len(momentum_factors)
    
    # Calculate the liquidity factor using the average volume over a dynamic lookback period
    liquidity_factors = [df['volume'].rolling(window=lookback).mean() for lookback in liquidity_lookbacks]
    liquidity = sum(liquidity_factors) / len(liquidity_factors)
    
    # Calculate the market sentiment using the average money flow ratio over a dynamic lookback period
    money_flow_ratio = df['amount'] / df['volume']
    market_sentiment_factors = [money_flow_ratio.rolling(window=lookback).mean() for lookback in market_sentiment_lookbacks]
    market_sentiment = sum(market_sentiment_factors) / len(market_sentiment_factors)
    
    # Calculate the volatility factor using the standard deviation of the log returns over a dynamic lookback period
    volatility_factors = [np.log(df['close']).diff().rolling(window=lookback).std() for lookback in volatility_lookbacks]
    volatility = sum(volatility_factors) / len(volatility_factors)
    
    # Calculate the intraday dynamics factor using the ratio of high to low prices over the last 5-10 days
    intraday_dynamics_factors = [(df['high'] / df['low']).rolling(window=lookback).mean() for lookback in intraday_dynamics_lookbacks]
    intraday_dynamics = sum(intraday_dynamics_factors) / len(intraday_dynamics_factors)
    
    # Normalize each factor by dividing by its mean to give equal weightage
    momentum_normalized = (momentum + 1) / (momentum + 1).mean()
    liquidity_normalized = liquidity / liquidity.mean()
    market_sentiment_normalized = market_sentiment / market_sentiment.mean()
    volatility_normalized = (volatility + 1e-6) / (volatility + 1e-6).mean()
    intraday_dynamics_normalized = intraday_dynamics / intraday_dynamics.mean()
    
    # Create a combined factor by multiplying the normalized factors
    # The idea is that stocks with positive momentum, high liquidity, strong market sentiment, lower volatility, and positive intraday dynamics may outperform
    factor = (momentum_normalized * liquidity_normalized * market_sentiment_normalized * intraday_dynamics_normalized) / volatility_normalized
    
    return factor
