import numpy as np
def heuristics(df):
    # Calculate Intraday Move
    df['IntradayMove'] = df['High'] - df['Close']
    
    # Calculate Intraday Volatility
    df['IntradayVolatility'] = df['High'] - df['Low']
    
    # Calculate Adjusted Daily Momentum
    df['DailyMomentum'] = df['Close'] - df['Close'].shift(1)
    df['AdjustedDailyMomentum'] = df['DailyMomentum'] / df['IntradayVolatility']
    
    # Estimate Trade Intensity
    df['TradeIntensity'] = df['Volume'] / ((df['High'] + df['Low']) / 2)
    
    # Weight Intraday Move by Trade Intensity
    df['WeightedIntradayMove'] = df['IntradayMove'] * df['TradeIntensity']
    
    # Weight Adjusted Daily Momentum by Trade Intensity
    df['WeightedAdjustedDailyMomentum'] = df['AdjustedDailyMomentum'] * df['TradeIntensity']
    
    # Calculate Intraday Reversal
    df['IntradayReversal'] = df['High'] - df['Open']
    
    # Weight Intraday Reversal by Trade Intensity
    df['WeightedIntradayReversal'] = df['IntradayReversal'] * df['TradeIntensity']
    
    # Calculate Intraday Gap
    df['IntradayGap'] = df['Open'] - df['Close'].shift(1)
    
    # Weight Intraday Gap by Trade Intensity
    df['WeightedIntradayGap'] = df['IntradayGap'] * df['TradeIntensity']
    
    # Calculate Intraday Midpoint
    df['IntradayMidpoint'] = (df['High'] + df['Low']) / 2
    
    # Calculate Intraday Range Ratio
    df['IntradayRangeRatio'] = (df['High'] - df['Low']) / df['IntradayMidpoint']
    
    # Weight Intraday Range Ratio by Trade Intensity
    df['WeightedIntradayRangeRatio'] = df['IntradayRangeRatio'] * df['TradeIntensity']
    
    # Calculate the 5-day and 21-day moving average of the close price
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA21'] = df['Close'].rolling(window=21).mean()
    
    # Compute the difference between the 5-day and 21-day moving averages
    df['TrendMomentum'] = df['MA5'] - df['MA21']
    
    # Calculate the 5-day and 21-day moving average of the volume
    df['VMA5'] = df['Volume'].rolling(window=5).mean()
    df['VMA21'] = df['Volume'].rolling(window=21).mean()
    
    # Compute the difference between the 5-day and 21-day moving averages of volume
    df['VolumeMomentum'] = df['VMA5'] - df['VMA21']
    
    # Calculate the 10-day simple moving average (SMA) of the high and low prices
    df['SMA10High'] = df['High'].rolling(window=10).mean()
    df['SMA10Low'] = df['Low'].rolling(window=10).mean()
    
    # Compute the difference between the 10-day SMA of high and low prices
    df['PriceOscillator'] = df['SMA10High'] - df['SMA10Low']
    
    # Calculate the Average True Range (ATR) over a 14-day period
    df['TrueRange'] = np.maximum(np.maximum(df['High'] - df['Low'], abs(df['High'] - df['Close'].shift(1))), abs(df['Low'] - df['Close'].shift(1)))
    df['ATR'] = df['TrueRange'].rolling(window=14).mean()
    
    # Calculate the 3-day RSI (Relative Strength Index) on the close price
    def rsi(close, periods=3):
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=periods).mean()
        avg_loss = loss.rolling(window=periods).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
