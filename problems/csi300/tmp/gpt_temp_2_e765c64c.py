import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate the True Range for each day
    df['PrevClose'] = df['close'].shift(1)
    df['TrueRange'] = df[['high', 'low']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - df.loc[x.name, 'PrevClose']), abs(x['low'] - df.loc[x.name, 'PrevClose'])), axis=1)

    # Calculate the 14-day Simple Moving Average of the True Range
    df['SMA_TR_14'] = df['TrueRange'].rolling(window=14).mean()

    # Construct the Momentum Component
    df['MomentumComponent'] = (df['close'] - df['SMA_TR_14']) / df['SMA_TR_14']

    # Calculate Intraday Reversal
    df['IntradayHighLowSpread'] = df['high'] - df['low']
    df['CloseToOpenReturn'] = df['close'] - df['open']
    df['IntradayReversal'] = (df['CloseToOpenReturn'] / df['IntradayHighLowSpread']) * df['IntradayHighLowSpread']

    # Incorporate Volume and Amount Influence
    df['AverageVolume_20'] = df['volume'].rolling(window=20).mean()
    df['IntradayVolumeImpact'] = df['volume'] / df['AverageVolume_20']
    df['AmountImpact'] = df['amount'] / df['AverageVolume_20']
    df['CombinedVolumeAmountImpact'] = df['IntradayVolumeImpact'] + df['AmountImpact']
    df['WeightedIntradayReversal'] = df['IntradayReversal'] * df['CombinedVolumeAmountImpact']

    # Calculate 14-Day Volume-Weighted Intraday Return
    df['DailyIntradayReturn'] = df['close'] - df['open']
    df['VolumeWeightedIntradayReturn_14'] = (df['DailyIntradayReturn'] * df['volume']).rolling(window=14).sum() / df['volume'].rolling(window=14).sum()

    # Calculate 14-Day Volume-Weighted Price Change
    df['DailyReturn'] = df['close'].pct_change()
    df['VolumeWeightedPriceChange_14'] = (df['DailyReturn'] * df['volume']).rolling(window=14).sum() / df['volume'].rolling(window=14).sum()

    # Enhance with Volume-Weighted High-Low Difference
    df['HighLowDifference'] = df['high'] - df['low']
    df['VolumeWeightedHighLowDifference'] = (df['HighLowDifference'] * df['volume']).rolling(window=14).sum() / df['volume'].rolling(window=14).sum()

    # Adjust for Volatility
    df['StdDev_30'] = df['close'].rolling(window=30).std()
    df['AdjustedMomentumComponent'] = df['MomentumComponent'] / df['StdDev_30']

    # Introduce Trend Component
    df['SMA_Close_50'] = df['close'].rolling(window=50).mean()
    df['TrendDirection'] = (df['close'] > df['SMA_Close_50']).astype(int) * 2 - 1

    # Synthesize Alpha Factor
    df['AlphaFactor'] = (
        df['AdjustedMomentumComponent'] +
        df['WeightedIntradayReversal'] +
        df['VolumeWeightedIntradayReturn_14'] +
        df['VolumeWeightedPriceChange_14'] +
        df['VolumeWeightedHighLowDifference']
    ) * df['TrendDirection']

    return df['AlphaFactor']
