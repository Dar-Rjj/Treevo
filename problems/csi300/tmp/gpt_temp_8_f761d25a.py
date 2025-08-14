import pandas as pd
import pandas as pd

def heuristics_v2(df):
    N = 10  # Number of days for the rolling window
    C = 100  # Constant for scaling momentum
    
    # Calculate the delta (t day close - (t-1) day close)
    df['delta'] = df['close'].diff()
    
    # Scale delta by volume and multiply by constant C
    df['scaled_momentum'] = (df['delta'] / df['volume']) * C
    
    # Initialize a column to store the rolling window of scaled momenta
    df['dynamic_momentum_indicator'] = 0.0
    
    # Initialize a list to store the past N scaled momenta
    past_scaled_momenta = []
    
    # Iterate over the DataFrame to calculate the dynamic momentum indicator
    for i, row in df.iterrows():
        if i > 0:
            # Append the new scaled momentum to the list
            past_scaled_momenta.append(row['scaled_momentum'])
            
            # Maintain a rolling window of the last N scaled momenta
            if len(past_scaled_momenta) > N:
                past_scaled_momenta.pop(0)
            
            # Assign weights (e.g., linearly decreasing)
            weights = [N - j for j in range(len(past_scaled_momenta))]
            total_weight = sum(weights)
            
            # Calculate the weighted sum of the past N scaled momenta
            weighted_sum = sum(m * w / total_weight for m, w in zip(past_scaled_momenta, weights))
            
            # Store the result in the DataFrame
            df.at[i, 'dynamic_momentum_indicator'] = weighted_sum
    
    # Return the dynamic momentum indicator as a pandas Series
    return df['dynamic_momentum_indicator']
