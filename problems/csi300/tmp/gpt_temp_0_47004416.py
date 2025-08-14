import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the Volume Weighted Average Price (VWAP)
    vwap = (df['amount'] / df['volume']).rolling(window=10).mean()
    
    # Calculate the Exponentially Weighted Moving Average (EWMA) of the close price
    ewma_close = df['close'].ewm(span=20, adjust=False).mean()
    
    # Calculate the log returns
    log_returns = np.log(df['close'] / df['close'].shift(1))
    
    # Momentum factor: 120-day moving average return
    momentum_120d = df['close'].pct_change(periods=120)
    
    # Volatility factor: 30-day standard deviation of log returns
    volatility_30d = log_returns.rolling(window=30).std()
    
    # Liquidity factor: 20-day mean of volume
    liquidity_20d = df['volume'].rolling(window=20).mean()
    
    # Trend factor: difference between today's close and the 120-day moving average
    trend_factor = df['close'] - df['close'].rolling(window=120).mean()
    
    # Relative strength index (RSI) as a measure of overbought/oversold conditions
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rsi = 100 - (100 / (1 + (gain / (loss + 1e-7))))
    
    # Seasonality factor: 5-day and 20-day rolling means
    seasonality_5d = df['close'].rolling(window=5).mean()
    seasonality_20d = df['close'].rolling(window=20).mean()
    seasonality_factor = (seasonality_5d - seasonality_20d) / (seasonality_20d + 1e-7)
    
    # Sentiment factor: hypothetical sentiment score (assuming it is available in the DataFrame)
    sentiment_score = df['sentiment'].rolling(window=10).mean()
    
    # Macroeconomic factor: hypothetical macroeconomic indicator (assuming it is available in the DataFrame)
    macro_indicator = df['macro_indicator'].rolling(window=30).mean()
    
    # Generate alpha factor as a weighted combination of the above metrics
    factor = (momentum_120d / (volatility_30d + 1e-7)) * (vwap / (ewma_close + 1e-7)) * (liquidity_20d / (df['volume'] + 1e-7)) * (trend_factor / (volatility_30d + 1e-7)) * (rsi / 50) * (seasonality_factor + 1) * (sentiment_score / 100) * (macro_indicator / 100)
    
    return factor
