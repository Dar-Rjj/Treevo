import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the difference between today's close and yesterday's close, capturing the momentum
    momentum = df['close'].diff()
    
    # Calculate the percentage change in volume from the previous day to detect unusual trading activity
    volume_change = df['volume'].pct_change().fillna(0)
    
    # Calculate the range (high - low) as a measure of volatility
    price_range = df['high'] - df['low']
    
    # Calculate the average true range (ATR) over a 14-day period to capture market depth
    atr = df[['high', 'low', 'close']].rolling(window=14).apply(lambda x: ta.volatility.average_true_range(x['high'], x['low'], x['close']), raw=False).fillna(0)
    
    # Calculate adaptive weights based on the ATR (higher ATR, lower weight on momentum, higher on volume and price range)
    adaptive_weights_momentum = 1 / (1 + atr)
    adaptive_weights_volume = 1 - adaptive_weights_momentum
    adaptive_weights_price_range = 0.5 * (adaptive_weights_volume + adaptive_weights_momentum)
    
    # Calculate a weighted sum of price and volume changes to capture significant market movements
    factor = (adaptive_weights_momentum * momentum) + (adaptive_weights_volume * (volume_change * df['amount'])) + (adaptive_weights_price_range * price_range)
    
    return factor
