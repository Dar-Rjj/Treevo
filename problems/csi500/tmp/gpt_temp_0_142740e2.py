import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Calculate Price Change
    df['price_change'] = df['close'].pct_change()
    
    # Calculate Volume Slope
    volume_slope = []
    for i in range(len(df)):
        if i < 5:
            slope = np.nan  # Not enough data points to calculate slope
        else:
            X = np.array(range(5)).reshape(-1, 1)
            y = df['volume'].iloc[i-4:i+1].values
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]
        volume_slope.append(slope)
    
    df['volume_slope'] = volume_slope
    
    # Combine Price Change and Volume Slope
    df['adjusted_price_change'] = df['price_change'] * df['volume_slope']
    
    # Add resulting value to Close price of current day (t) as a modifier
    df['factor_value'] = df['close'] + df['adjusted_price_change']
    
    return df['factor_value']
