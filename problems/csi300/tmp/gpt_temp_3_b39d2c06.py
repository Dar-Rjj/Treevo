import pandas as pd
import pandas as pd

def heuristics_v2(df):
    """
    This function calculates a novel and interpretable alpha factor that combines
    intraday range and close-to-open return with dynamic weighting based on the recency of the data.
    """
    # Calculate Intraday Range
    df['intraday_range'] = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = (df['close'] - df['open']) / df['open']
    
    # Define dynamic weights based on the recency of the data
    df['days_since_start'] = (df.index - df.index[0]).days
    recent_data_mask = df['days_since_start'] > df['days_since_start'].quantile(0.5)
    
    df['intraday_weight'] = 0.7 * recent_data_mask + 0.5 * ~recent_data_mask
    df['close_to_open_weight'] = 0.3 * recent_data_mask + 0.5 * ~recent_data_mask
    
    # Combine Intraday Range and Close-to-Open Return with dynamic weights
    df['alpha_factor'] = (df['intraday_range'] * df['intraday_weight']) + \
                         (df['close_to_open_return'] * df['close_to_open_weight'])
    
    # Return the alpha factor as a pandas Series
    return df['alpha_factor']

# Example usage:
# Assuming `df` is a pandas DataFrame with the required columns
# alpha_factor_series = heuristics_v2(df)
