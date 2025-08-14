import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Volume-Weighted Price Returns
    df['NextDayOpen'] = df['open'].shift(-1)
    df['SimpleReturns'] = (df['NextDayOpen'] - df['close']) / df['close']
    df['VolumeWeightedReturns'] = df['SimpleReturns'] * df['volume']

    # Identify Volume Surge Days
    df['VolumeChange'] = df['volume'] - df['volume'].shift(1)
    df['VolumeRollingMean'] = df['volume'].rolling(window=5).mean()
    df['VolumeSurge'] = (df['volume'] > df['VolumeRollingMean']).astype(int)

    # Calculate Volatility Using ATR
    df['TrueRange'] = df.apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift(1)), abs(x['low'] - x['close'].shift(1))), axis=1)
    df['ATR'] = df['TrueRange'].rolling(window=14).mean()

    # Adjust Volume-Weighted Returns by ATR
    df['AdjustedReturns'] = df['VolumeWeightedReturns'] / df['ATR']

    # Combine Adjusted Returns with Volume Surge Indicator
    surge_factor = 1.5
    df['AdjustedReturns'] = df['AdjustedReturns'] * (1 + (surge_factor - 1) * df['VolumeSurge'])

    # Incorporate Short-Term Trend
    df['5DayMA'] = df['close'].rolling(window=5).mean()
    df['5DayMomentum'] = (df['close'] - df['5DayMA']) / df['5DayMA']

    # Combine All Factors
    weight_adjusted_returns = 0.7
    weight_5day_momentum = 0.3
    df['AlphaFactor'] = (weight_adjusted_returns * df['AdjustedReturns'] + 
                         weight_5day_momentum * df['5DayMomentum'])

    return df['AlphaFactor']
