import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['NextDayOpen'] = df['open'].shift(-1)
    df['CloseToOpenReturn'] = df['NextDayOpen'] - df['close']
    
    # Volume Weighting
    df['VolumeWeightedCO'] = df['Volume'] * df['CloseToOpenReturn']
    
    # Incorporate Trading Range
    df['HighLowDiff'] = df['high'] - df['low']
    df['HLAdjustedVolumeWeightedCO'] = df['VolumeWeightedCO'] * df['HighLowDiff']
    
    # Incorporate Amount
    df['AmountAdjustedVolumeWeightedCO'] = df['HLAdjustedVolumeWeightedCO'] * df['amount']
    
    # Adaptive Window Calculation
    fixed_window = 20
    df['Volatility'] = df[['high', 'low', 'close']].rolling(window=fixed_window).std().mean(axis=1)
    volatility_threshold = df['Volatility'].median()
    
    def adjust_window(volatility, threshold, high_window=40, low_window=10):
        if volatility > threshold:
            return min(high_window, int(low_window * (volatility / threshold)))
        else:
            return max(low_window, int(high_window * (threshold / volatility)))
    
    df['WindowSize'] = df['Volatility'].apply(lambda v: adjust_window(v, volatility_threshold))
    
    # Rolling Statistics
    def rolling_stats(series, window_series):
        return series.rolling(window=window_series, min_periods=1).mean(), \
               series.rolling(window=window_series, min_periods=1).std()
    
    df['AdaptiveWindowMean'], df['AdaptiveWindowStd'] = zip(*df.apply(lambda row: rolling_stats(pd.Series([row['AmountAdjustedVolumeWeightedCO']]), row['WindowSize']), axis=1))
    
    # Factor
    factor = df['AdaptiveWindowMean'] / df['AdaptiveWindowStd']
    
    return factor.dropna()

# Example usage
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
