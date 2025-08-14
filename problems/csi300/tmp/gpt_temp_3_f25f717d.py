import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Volume-Weighted Price Returns
    df['NextDayOpen'] = df['open'].shift(-1)
    df['SimpleReturns'] = (df['NextDayOpen'] - df['close']) / df['close']
    df['VolumeWeightedReturns'] = df['SimpleReturns'] * df['volume']

    # Identify Volume Surge Days
    df['VolumeChange'] = df['volume'] - df['volume'].shift(1)
    df['RollingMeanVolume'] = df['volume'].rolling(window=5).mean()
    df['VolumeSurge'] = (df['volume'] > df['RollingMeanVolume']).astype(int)

    # Calculate Enhanced Volatility
    df['DailyReturns'] = df['close'].pct_change()
    df['StdDevDailyReturns'] = df['DailyReturns'].rolling(window=20).std()
    df['VolumeMA'] = df['volume'].rolling(window=20).mean()
    df['VolumeStdDev'] = df['volume'].rolling(window=20).std()
    df['VolumeZScore'] = (df['volume'] - df['VolumeMA']) / df['VolumeStdDev']
    df['EnhancedVolatility'] = df['StdDevDailyReturns'] * (1 + abs(df['VolumeZScore']))

    # Incorporate Sector Performance
    # Assuming 'sector_close' is a column in the DataFrame representing the close price of the sector index
    df['SectorReturns'] = df['sector_close'].pct_change()
    df['SectorAlpha'] = df['DailyReturns'] - df['SectorReturns']
    df['AdjustedVolumeWeightedReturns'] = df['VolumeWeightedReturns'] * df['SectorAlpha']

    # Adjust Volume-Weighted Returns by Enhanced Volatility
    df['VolatilityAdjustedReturns'] = df['AdjustedVolumeWeightedReturns'] / df['EnhancedVolatility']

    # Incorporate Market Sentiment
    # Assuming 'news_sentiment' is a column in the DataFrame representing the news sentiment score
    df['MarketSentiment'] = df['news_sentiment'].apply(
        lambda x: 1.2 if x > 0 else (0.8 if x < 0 else 1)
    )
    df['SentimentAdjustedReturns'] = df['VolatilityAdjustedReturns'] * df['MarketSentiment']

    # Incorporate Adaptive Surge Factors
    df['VolumeSurgeRatio'] = df['volume'] / df['volume'].shift(1)
    df['AdaptiveSurgeFactor'] = df['VolumeSurgeRatio'].apply(
        lambda x: 1.7 if x > 2.0 else (1.3 if x > 1.5 else 1)
    )
    df['FinalAlphaFactor'] = df['SentimentAdjustedReturns'] * df['AdaptiveSurgeFactor']

    return df['FinalAlphaFactor']

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
