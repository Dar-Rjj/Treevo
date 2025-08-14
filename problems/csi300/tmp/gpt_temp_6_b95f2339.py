import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Volume-Weighted Close to Next Day Open Return
    df['NextDayOpen'] = df['open'].shift(-1)
    df['CloseToNextOpenReturn'] = (df['NextDayOpen'] - df['close']) / df['close']
    df['VolumeWeightedReturn'] = df['CloseToNextOpenReturn'] * df['volume']

    # Identify Volume Surge Days
    df['VolumeChange'] = df['volume'] - df['volume'].shift(1)
    df['VolumeRollingMean'] = df['volume'].rolling(window=5).mean()
    df['IsVolumeSurge'] = df['volume'] > df['VolumeRollingMean']
    
    # Incorporate Recent Volatility
    df['20DayVolatility'] = df['close'].rolling(window=20).std()
    df['AdjustedMomentum'] = df['VolumeWeightedReturn'] / df['20DayVolatility']

    # Integrate Market Sentiment
    df['IntradayHighLowSpreadRatio'] = (df['high'] - df['low']) / df['low']
    df['PriceChange'] = df['close'] - df['open']
    df['VolumeWeightedPriceChange'] = df['PriceChange'] * df['volume']
    df['14DayHLVolatility'] = df[['high', 'low', 'close']].apply(lambda x: np.std(x, ddof=0), axis=1).rolling(window=14).std()
    df['10DayVolumeTrend'] = df['volume'].rolling(window=10).mean()
    
    # Combine Intraday Metrics with Dynamic Weights
    df['DynamicWeight'] = 0.5 * (df['14DayHLVolatility'] / df['14DayHLVolatility'].max()) + 0.5 * (df['10DayVolumeTrend'] / df['10DayVolumeTrend'].max())
    df['MarketSentiment'] = df['IntradayHighLowSpreadRatio'] * df['DynamicWeight'] + df['VolumeWeightedPriceChange'] * (1 - df['DynamicWeight'])

    # Combine Weighted Returns with Volume Surge Indicator and Market Sentiment
    df['FinalAlphaFactor'] = df['AdjustedMomentum']
    df.loc[df['IsVolumeSurge'], 'FinalAlphaFactor'] *= 1.5
    df['FinalAlphaFactor'] *= df['MarketSentiment']

    return df['FinalAlphaFactor'].dropna()
