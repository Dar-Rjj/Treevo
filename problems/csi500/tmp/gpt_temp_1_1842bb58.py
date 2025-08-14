import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily VWAP
    df['TotalVolume'] = df['volume']
    df['DollarValue'] = (df['high'] + df['low'] + df['close']) / 3 * df['volume']
    df['DailyVWAP'] = df['DollarValue'].rolling(window=1, min_periods=1).sum() / df['TotalVolume'].rolling(window=1, min_periods=1).sum()

    # Calculate VWAP Deviation
    df['VWAPDeviation'] = df['close'] - df['DailyVWAP']

    # Calculate Cumulative VWAP Deviation
    df['CumulativeVWAPDeviation'] = df.groupby(df.index).apply(lambda x: x['VWAPDeviation'].cumsum()).droplevel(0)

    # Integrate Intraday Momentum Intensity
    df['IntradayRangePercentage'] = (df['high'] - df['low']) / df['low']
    df['IntradayMomentumIntensity'] = df['volume'] * df['IntradayRangePercentage']
    df['AdjustedCumulativeVWAPDeviation'] = df['CumulativeVWAPDeviation'] * df['IntradayMomentumIntensity']

    # Incorporate Multi-Period Momentum
    df['5DayCloseReturn'] = df['close'].pct_change(periods=5)
    df['20DayCloseReturn'] = df['close'].pct_change(periods=20)
    
    # Dynamic Weighting Based on Recent Performance
    df['30DayHistoricalPerformance'] = df['close'].pct_change(periods=30).fillna(0)
    df['30DayMarketVolatility'] = df['close'].rolling(window=30).std().fillna(0)
    
    # Determine Weights Based on Performance and Adjust for Volatility
    df['WeightedPerformance'] = df['30DayHistoricalPerformance'] * (1 - df['30DayMarketVolatility'])
    
    # Apply Weights to Final Alpha Factor
    df['FinalAlphaFactor'] = (
        df['AdjustedCumulativeVWAPDeviation'] * df['WeightedPerformance'] +
        df['5DayCloseReturn'] * df['WeightedPerformance'] +
        df['20DayCloseReturn'] * df['WeightedPerformance']
    )

    return df['FinalAlphaFactor']

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df)
