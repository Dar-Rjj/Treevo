import pandas as pd
import pandas as pd

def heuristics(df):
    # Calculate Daily Momentum
    df['DailyMomentum'] = df['close'].diff()

    # Identify Volume Spikes
    df['20DayVolMA'] = df['volume'].rolling(window=20).mean()
    df['VolumeSpike'] = (df['volume'] > 1.5 * df['20DayVolMA']).astype(int)

    # Adjust Daily Momentum by Volume Spike
    df['AdjustedMomentum'] = df['DailyMomentum'] * (2 * df['VolumeSpike'] + 1 - df['VolumeSpike'])

    # Combine High-Low Range and Close-to-Open Return
    df['HighLowRange'] = df['high'] - df['low']
    df['CloseOpenReturn'] = df['close'] - df['open']
    df['CombinedFactor'] = df['HighLowRange'] + df['CloseOpenReturn']

    # Adjust by Volume
    df['AdjustedCombinedFactor'] = df['CombinedFactor'] / df['20DayVolMA']

    # Calculate 14-Period Exponential Moving Averages
    df['HighEMA'] = df['high'].ewm(span=14, adjust=False).mean()
    df['LowEMA'] = df['low'].ewm(span=14, adjust=False).mean()
    df['CloseEMA'] = df['close'].ewm(span=14, adjust=False).mean()
    df['OpenEMA'] = df['open'].ewm(span=14, adjust=False).mean()

    # Compute 14-Period Price Envelopes
    df['MaxPrice'] = df[['high', 'close']].max(axis=1)
    df['MinPrice'] = df[['low', 'close']].min(axis=1)
    df['MaxPriceEMA'] = df['MaxPrice'].ewm(span=14, adjust=False).mean()
    df['MinPriceEMA'] = df['MinPrice'].ewm(span=14, adjust=False).mean()
    df['EnvelopeDistance'] = df['MaxPriceEMA'] - df['MinPriceEMA']
    df['VolumeSmoothedEnvelope'] = (df['EnvelopeDistance'] * df['volume']).rolling(window=14).mean()

    # Construct Momentum Oscillator
    df['PositiveMomentum'] = (df['HighEMA'] - df['CloseEMA']) * df['VolumeSmoothedEnvelope']
    df['NegativeMomentum'] = (df['LowEMA'] - df['CloseEMA']) * df['VolumeSmoothedEnvelope']
    df['PositiveMomentum'] = df['PositiveMomentum'].apply(lambda x: x if x > 0 else 0)
    df['NegativeMomentum'] = df['NegativeMomentum'].apply(lambda x: x if x < 0 else 0)
    df['MomentumIndicator'] = df['PositiveMomentum'] - df['NegativeMomentum']

    # Introduce Volatility Adjustment
    df['TrueRange'] = df[['high' - df['low'], abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))]].max(axis=1)
    df['ATR'] = df['TrueRange'].rolling(window=20).mean()
    df['VolatilityAdjustedFactor'] = df['AdjustedCombinedFactor'] / df['ATR']

    # Final Alpha Factor
    df['AlphaFactor'] = 0.7 * df['VolatilityAdjustedFactor'] + 0.3 * df['MomentumIndicator']

    return df['AlphaFactor']
