import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate momentum factor by taking the 30 day difference in close price to incorporate longer-term trends
    momentum_factor = df['close'].diff(30)
    
    # Calculate 5-day and 20-day moving average of close price for trend analysis
    ma_5d = df['close'].rolling(window=5).mean()
    ma_20d = df['close'].rolling(window=20).mean()
    
    # Calculate the ratio of the 5-day MA to the 20-day MA as a smoothed indicator of the short-term trend relative to the long-term trend
    trend_ratio = ma_5d / ma_20d
    
    # Calculate volatility over a 10-day window
    volatility_factor = df['close'].rolling(window=10).std()
    
    # Calculate trading volume intensity as a percentage of the average volume over a 20-day period
    avg_volume_20d = df['volume'].rolling(window=20).mean()
    volume_intensity = (df['volume'] - avg_volume_20d) / avg_volume_20d

    # Calculate the range between high and low prices to measure daily volatility
    daily_range = df['high'] - df['low']
    
    # Calculate the average true range (ATR) over a 14-day period to measure market volatility
    tr = df[['high', 'low']].apply(lambda x: max(x['high'], df['close'].shift()) - min(x['low'], df['close'].shift()), axis=1)
    atr = tr.rolling(window=14).mean()

    # Generate the alpha factor as a weighted sum of these components, with weights chosen to balance their contributions
    alpha_factor = (momentum_factor * 0.3) + (trend_ratio * 0.2) + (volatility_factor * 0.1) + (volume_intensity * 0.2) + (atr * 0.2)

    return alpha_factor
