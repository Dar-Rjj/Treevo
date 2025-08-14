import numpy as np
def heuristics_v2(df):
    # Calculate Momentum
    df['ShortTermMomentum'] = df['close'].pct_change(10)
    df['MediumTermMomentum'] = df['close'].pct_change(30)
    df['LongTermMomentum'] = df['close'].pct_change(90)

    # Calculate Intraday Range
    df['IntradayRange'] = df['high'] - df['low']

    # Calculate Price Change
    df['PriceChange'] = df['close'].diff()

    # Detect Volume Spikes
    avg_volume = df['volume'].rolling(window=20).mean()
    df['VolumeSpike'] = (df['volume'] > 2 * avg_volume).astype(int)

    # Adjust Price Change by Intraday Range
    df['AdjustedPriceChange'] = df['PriceChange'] / df['IntradayRange']

    # Weight by Volume
    df['WeightedByVolume'] = df['volume'] * df['AdjustedPriceChange']

    # Enhance Momentum on Volume Spike Days
    df['EnhancedMomentum'] = df['WeightedByVolume']
    df.loc[df['VolumeSpike'] == 1, 'EnhancedMomentum'] *= 2

    # Cumulative Momentum
    df['CumulativeMomentum'] = df['EnhancedMomentum'].rolling(window=60).sum()

    # Adjust by Volatility and Trading Volume
    volatility = df['close'].rolling(window=20).std()
    avg_trading_volume = df['volume'].rolling(window=30).mean()
    df['ComplexityScore'] = (df['CumulativeMomentum'] / volatility) * avg_trading_volume

    # Compute Daily Log Return
    df['DailyLogReturn'] = np.log(df['close']) - np.log(df['close'].shift(1))

    # Calculate Volume-Weighted Exponential Moving Average (VWEMA) of Returns
    def vwema(data, window, vol):
        return (data * vol).ewm(span=window, adjust=False).mean() / vol.ewm(span=window, adjust=False).mean()
