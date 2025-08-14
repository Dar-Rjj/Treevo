import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Return
    df['IntradayRange'] = df['high'] - df['low']
    df['IntradayReturn'] = df['IntradayRange'] / df['close']

    # Calculate Overnight Return
    df['OvernightReturn'] = (df['open'].shift(-1) - df['close']) / df['close']

    # Combine Intraday and Overnight Returns
    df['CombinedReturn'] = df['IntradayReturn'] + df['OvernightReturn']

    # Compute Volume Weighted Average Price (VWAP)
    df['TypicalPrice'] = (df['high'] + df['low'] + df['close']) / 3
    df['VolumeWeightedPrice'] = df['TypicalPrice'] * df['volume']
    total_volume = df['volume'].sum()
    vwap = df['VolumeWeightedPrice'].sum() / total_volume
    df['VWAP'] = vwap

    # Calculate VWAP Reversal Indicator
    df['ReversalIndicator'] = df.apply(lambda row: 1 if row['VWAP'] > row['close'] else -1, axis=1)

    # Integrate Reversal Indicator with Combined Return
    df['IntegratedReversalReturn'] = df['CombinedReturn'] * df['ReversalIndicator']

    # Calculate Volume-Weighted Return
    df['VolumeWeightedReturn'] = ((df['close'] - df['open']) / df['open']) * df['volume']
    volume_weighted_return = df['VolumeWeightedReturn'].sum() / total_volume
    df['VolumeWeightedReturn'] = volume_weighted_return

    # Integrate Volume-Weighted Return with Integrated Reversal Return
    df['FinalAlphaFactor'] = df['IntegratedReversalReturn'] + df['VolumeWeightedReturn']

    # Short-Term Volatility
    df['TrueRange'] = df[['high', 'low', 'close']].apply(
        lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift(1)), abs(x['low'] - x['close'].shift(1))), 
        axis=1
    )
    short_term_volatility = df['TrueRange'].rolling(window=5).mean()
    df['ShortTermVolatility'] = short_term_volatility

    # Integrate Short-Term Volatility with Final Alpha Factor
    threshold = 0.05  # Example threshold value
    df['FinalAlphaFactor'] = df.apply(
        lambda row: row['FinalAlphaFactor'] * 0.9 if row['ShortTermVolatility'] > threshold else row['FinalAlphaFactor'],
        axis=1
    )

    return df['FinalAlphaFactor']
