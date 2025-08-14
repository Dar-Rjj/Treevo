import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate the difference between high and low prices as a measure of daily volatility
    df['daily_volatility'] = df['high'] - df['low']
    
    # Evaluate the body length (difference between open and close) as an indicator of directional movement
    df['body_length'] = abs(df['close'] - df['open'])
    
    # Identify doji patterns (where open and close are approximately equal) indicating indecision in the market
    df['doji'] = (df['open'] == df['close']).astype(int)
    
    # Detect hammer and hanging man patterns to signal potential trend reversals
    df['hammer_hanging_man'] = (
        (2 * df['close'] - df['high'] - df['low']) > 0.3 * (df['high'] - df['low']) &
        (abs(df['close'] - df['open']) / (df['high'] - df['low']) < 0.1)
    ).astype(int)
    
    # Compute the ratio of today's volume to the average volume over a lookback period to find unusual trading activity
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=10).mean()
    
    # Measure the correlation between closing price change and volume change to assess buying or selling pressure
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['corr_price_volume'] = df[['price_change', 'volume_change']].rolling(window=10).corr().loc[(slice(None), 'price_change'), 'volume_change']
    
    # Analyze the effect of consecutive days of increasing volume on future returns
    df['consecutive_increasing_volume'] = (df['volume'] > df['volume'].shift(1)).rolling(window=3).sum()
    
    # Develop a short-term moving average crossover system (e.g., 5-day vs 10-day) to detect trend changes
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_10'] = df['close'].rolling(window=10).mean()
    df['crossover_signal'] = (df['ma_5'] > df['ma_10']).astype(int) - (df['ma_5'] < df['ma_10']).astype(int)
    
    # Utilize the relative position of the closing price compared to the day's range (close - open / high - low) to quantify bullish or bearish sentiment
    df['close_position'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    
    # Create a simple oscillator using the difference between a short-term and long-term moving average to identify overbought or oversold conditions
    df['short_ma'] = df['close'].rolling(window=5).mean()
    df['long_ma'] = df['close'].rolling(window=20).mean()
    df['oscillator'] = df['short_ma'] - df['long_ma']
    
    # Incorporate a weighted moving average that gives more weight to recent prices to enhance trend detection
    df['weighted_ma'] = df['close'].ewm(span=5, adjust=False).mean()
    
    # Combine all factors into a single alpha factor
    alpha_factor = (
        df['daily_volatility'] * 0.1 +
        df['body_length'] * 0.1 +
        df['doji'] * 0.1 +
        df['hammer_hanging_man'] * 0.1 +
        df['volume_ratio'] * 0.1 +
        df['corr_price_volume'] * 0.1 +
        df['consecutive_increasing_volume'] * 0.1 +
        df['crossover_signal'] * 0.1 +
        df['close_position'] * 0.1 +
        df['oscillator'] * 0.1 +
        df['weighted_ma'] * 0.1
    )
    
    return alpha_factor
