import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Return
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    
    # Calculate Overnight Return
    df['overnight_return'] = (df['open'].shift(-1) - df['close']) / df['close']
    
    # Combine Returns
    df['combined_return'] = (df['overnight_return'] - df['intraday_return']) * df['volume']
    
    # Calculate Rolling Sum of Volumetric Weighted Returns
    # Set an initial lookback period
    lookback_period = 20
    
    # Calculate Average Volume
    df['average_volume'] = df['volume'].rolling(window=lookback_period).mean()
    
    # Adjust Lookback Based on Market Activity
    # Define thresholds for high and low activity
    high_activity_threshold = 1.5 * df['average_volume'].mean()
    low_activity_threshold = 0.5 * df['average_volume'].mean()
    
    # Adjust lookback period based on market activity
    def adjust_lookback(average_volume, high_activity_threshold, low_activity_threshold, lookback_period):
        if average_volume > high_activity_threshold:
            return max(5, lookback_period - 5)
        elif average_volume < low_activity_threshold:
            return min(50, lookback_period + 5)
        else:
            return lookback_period
    
    df['adjusted_lookback'] = df.apply(lambda row: adjust_lookback(row['average_volume'], high_activity_threshold, low_activity_threshold, lookback_period), axis=1)
    
    # Use the adjusted lookback to calculate the rolling sum of volumetric weighted returns
    df['volumetric_weighted_momentum'] = df.groupby(df.index)['combined_return'].transform(lambda x: x.rolling(window=x.name[1]['adjusted_lookback']).sum())
    
    # Return the factor values
    return df['volumetric_weighted_momentum']

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df)
