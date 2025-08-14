import pandas as pd
import pandas as pd

def heuristics_v2(df, n=5):
    # Calculate the difference between today's close and open prices
    price_diff = df['close'] - df['open']
    
    # Calculate the range (High - Low) as a measure of intraday volatility
    intraday_volatility = df['high'] - df['low']
    
    # Compute the ratio of current day's volume to the average volume over the past n days
    avg_volume = df['volume'].rolling(window=n).mean()
    volume_ratio = df['volume'] / avg_volume
    
    # Calculate the simple moving average (SMA) of the closing prices over the past n days
    sma_close = df['close'].rolling(window=n).mean()
    
    # Determine the rate of change (ROC) of the closing price from the previous day
    roc_close = df['close'].pct_change()
    
    # Calculate the ratio of today's close to the average close over the past n days
    avg_close = df['close'].rolling(window=n).mean()
    close_ratio = df['close'] / avg_close
    
    # Find the correlation between the daily amount and the closing prices over the past n days
    corr_amount_close = df[['amount', 'close']].rolling(window=n).corr().dropna()
    corr_amount_close = corr_amount_close.xs('amount', level=1, axis=1)['close']
    
    # Combine the factors into a single alpha factor
    alpha_factor = (price_diff + intraday_volatility + volume_ratio + 
                    sma_close + roc_close + close_ratio + corr_amount_close)
    
    # Normalize the alpha factor
    alpha_factor = (alpha_factor - alpha_factor.mean()) / alpha_factor.std()
    
    return alpha_factor
