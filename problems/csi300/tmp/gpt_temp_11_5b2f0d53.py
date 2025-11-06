import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum Decay with Asymmetric Volatility Adjustment alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Price Momentum components
    data['short_term_return'] = data['close'] / data['close'].shift(2) - 1
    data['medium_term_return'] = data['close'] / data['close'].shift(5) - 1
    
    # Apply Exponential Decay to momentum
    decay_factor = 0.9  # Higher weight to recent momentum
    data['decayed_momentum'] = (data['short_term_return'] * decay_factor + 
                               data['medium_term_return'] * (1 - decay_factor))
    
    # Calculate Asymmetric Volatility
    returns = data['close'].pct_change()
    
    # Calculate Upside Volatility (t-20 to t-1)
    upside_vol = pd.Series(index=data.index, dtype=float)
    for i in range(20, len(data)):
        window_returns = returns.iloc[i-20:i]
        positive_returns = window_returns[window_returns > 0]
        if len(positive_returns) > 1:
            upside_vol.iloc[i] = positive_returns.std()
        else:
            upside_vol.iloc[i] = 0
    
    # Calculate Downside Volatility (t-20 to t-1)
    downside_vol = pd.Series(index=data.index, dtype=float)
    for i in range(20, len(data)):
        window_returns = returns.iloc[i-20:i]
        negative_returns = window_returns[window_returns < 0]
        if len(negative_returns) > 1:
            downside_vol.iloc[i] = negative_returns.std()
        else:
            downside_vol.iloc[i] = 0
    
    # Compute Volatility Ratio
    data['volatility_ratio'] = upside_vol / downside_vol.replace(0, np.nan)
    data['volatility_ratio'] = data['volatility_ratio'].fillna(1)  # Handle division by zero
    
    # Volume-Weighted Confirmation
    data['avg_volume_10d'] = data['volume'].rolling(window=10, min_periods=1).mean()
    data['volume_ratio'] = data['volume'] / data['avg_volume_10d'].replace(0, np.nan)
    data['volume_ratio'] = data['volume_ratio'].fillna(1)
    
    # Combine all components
    alpha_factor = (data['decayed_momentum'] * 
                   data['volatility_ratio'] * 
                   data['volume_ratio'])
    
    return alpha_factor
