import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Intraday Momentum
    momentum = (df['high'] - df['low']) / (df['high'] + df['low'])
    
    # Calculate Rolling Volatility
    volatility = df['close'].rolling(window=5).std()
    
    # Apply Exponential Decay Weighting
    decay_factor = 2 ** (-1 / 3)  # 3-day half-life
    weights = [decay_factor ** i for i in range(len(df))]
    weights = weights[::-1]  # Reverse to give more weight to recent values
    
    # Multiply Momentum by Volatility and apply decay weights
    momentum_vol = momentum * volatility
    factor_values = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i >= 4:  # Need at least 5 days for volatility calculation
            weighted_sum = 0
            weight_sum = 0
            for j in range(i + 1):
                if not pd.isna(momentum_vol.iloc[j]):
                    weight = weights[i - j]
                    weighted_sum += momentum_vol.iloc[j] * weight
                    weight_sum += weight
            factor_values.iloc[i] = weighted_sum / weight_sum if weight_sum > 0 else 0
        else:
            factor_values.iloc[i] = 0
    
    return factor_values
