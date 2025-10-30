import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Price Acceleration
    # Short-Term Momentum (3-day)
    short_term_momentum = data['close'] - data['close'].shift(3)
    
    # Medium-Term Momentum (10-day)
    medium_term_momentum = data['close'] - data['close'].shift(10)
    
    # Acceleration (change in momentum speed)
    acceleration = short_term_momentum - medium_term_momentum
    
    # Calculate Volume Divergence
    # Compute returns and volume changes over 10-day window
    returns = data['close'].pct_change()
    volume_changes = data['volume'].pct_change()
    
    # Initialize divergence signal
    divergence_signal = pd.Series(0, index=data.index)
    
    # Calculate rolling correlation between price returns and volume changes
    for i in range(10, len(data)):
        # Use only historical data from t-10 to t
        window_returns = returns.iloc[i-10:i+1]
        window_volume = volume_changes.iloc[i-10:i+1]
        
        # Calculate correlation using only available data
        if len(window_returns.dropna()) >= 5 and len(window_volume.dropna()) >= 5:
            corr = window_returns.corr(window_volume)
            
            # Detect divergence patterns
            current_return = returns.iloc[i]
            current_volume_change = volume_changes.iloc[i]
            
            if current_return > 0 and current_volume_change < 0:
                divergence_signal.iloc[i] = -1  # Positive price change with decreasing volume
            elif current_return < 0 and current_volume_change > 0:
                divergence_signal.iloc[i] = 1   # Negative price change with increasing volume
            else:
                divergence_signal.iloc[i] = 0   # No clear divergence
    
    # Combine components
    alpha_factor = acceleration * divergence_signal
    
    # Handle NaN values by forward filling (using only past information)
    alpha_factor = alpha_factor.fillna(method='ffill')
    
    return alpha_factor
