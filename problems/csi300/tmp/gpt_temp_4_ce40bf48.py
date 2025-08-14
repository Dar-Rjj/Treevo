import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure the DataFrame is sorted by date
    df = df.sort_index()
    
    # Calculate Intraday Range (High - Low)
    intraday_range = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    close_to_open_return = (df['close'] - df['open']) / df['open']
    
    # Calculate Momentum
    momentum = df['close'].pct_change(periods=1)
    
    # Function to calculate dynamic weights
    def calculate_weights(volume, momentum, recent_data=True):
        if recent_data:
            base_intraday_weight = 0.7
            base_close_to_open_weight = 0.3
        else:
            base_intraday_weight = 0.5
            base_close_to_open_weight = 0.5
        
        # Adjust weights based on volume
        if volume > df['volume'].mean():
            intraday_weight = base_intraday_weight * 1.2
            close_to_open_weight = base_close_to_open_weight * 0.8
        else:
            intraday_weight = base_intraday_weight * 0.8
            close_to_open_weight = base_close_to_open_weight * 1.2
        
        # Adjust weights based on momentum
        if momentum > 0:
            close_to_open_weight *= 1.2
            intraday_weight *= 0.8
        else:
            close_to_open_weight *= 0.8
            intraday_weight *= 1.2
        
        return intraday_weight, close_to_open_weight
    
    # Initialize the factor values
    factor_values = []
    
    for i in range(len(df)):
        if i == 0:
            # For the first day, use equal weights
            intraday_weight, close_to_open_weight = 0.5, 0.5
        else:
            # Calculate weights based on volume and momentum
            intraday_weight, close_to_open_weight = calculate_weights(df.iloc[i]['volume'], momentum.iloc[i], recent_data=(i > 1))
        
        # Combine Intraday Range and Close-to-Open Return
        factor_value = intraday_weight * intraday_range.iloc[i] + close_to_open_weight * close_to_open_return.iloc[i]
        factor_values.append(factor_value)
    
    # Create a pandas Series with the calculated factor values
    factor_series = pd.Series(factor_values, index=df.index, name='intraday_volatility_adjusted_return')
    
    return factor_series
