import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate 10-day Average Price Range
    df['PriceRange'] = df['high'] - df['low']
    df['10DayAvgPriceRange'] = df['PriceRange'].rolling(window=10).mean()

    # Compute Volume-Adjusted Momentum
    df['Momentum'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    df['VolumeAdjustedMomentum'] = df['Momentum'] * df['volume']

    # Calculate Adjusted Intraday High-Low Spread with Directional Bias
    df['IntradayRange'] = df['high'] - df['low']
    df['VolumeWeightedIntradayRange'] = df['IntradayRange'] * df['volume']
    df['DirectionalBias'] = df['VolumeWeightedIntradayRange'] * (1 if df['close'] > df['open'] else -1)

    # Calculate Volume-Adjusted Opening Gap
    df['OpeningGap'] = df['open'] - df['close'].shift(1)
    df['VolumeAdjustedOpeningGap'] = df['OpeningGap'] * df['volume']

    # Combine Weighted Intraday High-Low Spread with Volume-Adjusted Opening Gap
    df['CombinedIndicator'] = df['VolumeWeightedIntradayRange'] + df['VolumeAdjustedOpeningGap']

    # Calculate Short-Term and Long-Term Moving Averages
    df['5DayMA'] = df['close'].rolling(window=5).mean()
    df['20DayMA'] = df['close'].rolling(window=20).mean()

    # Compute Crossover Signal
    df['CrossoverSignal'] = df['5DayMA'] - df['20DayMA']

    # Generate Alpha Factor Based on Crossover
    df['AlphaFactor'] = df['CrossoverSignal'].apply(lambda x: 1 if x > 0 else -1)

    # Integrate Combined Indicator and Alpha Factor
    df['IntegratedValue'] = df.apply(
        lambda row: row['CombinedIndicator'] + row['AlphaFactor'] if row['AlphaFactor'] == 1 else row['CombinedIndicator'] - row['AlphaFactor'],
        axis=1
    )

    # Integrate Volume-Adjusted Momentum
    df['IntegratedValue'] = df['IntegratedValue'] + df['VolumeAdjustedMomentum']

    # Consider Directional Bias
    df['FinalAlphaFactor'] = df.apply(
        lambda row: row['IntegratedValue'] * 1.5 if row['close'] > row['open'] else row['IntegratedValue'] * 0.5,
        axis=1
    )

    return df['FinalAlphaFactor']
