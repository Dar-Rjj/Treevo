import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Initialize result series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    # Calculate short-term momentum components (5-day)
    short_price_momentum = np.log(data['close'] / data['close'].shift(5))
    short_volume_momentum = data['volume'] / data['volume'].shift(5)
    
    # Apply exponential decay for short-term (5-day window)
    short_weights = np.exp(-0.2 * np.arange(5, 0, -1))
    short_weights /= short_weights.sum()
    
    # Calculate weighted short-term momentum
    short_price_weighted = pd.Series(index=data.index, dtype=float)
    short_volume_weighted = pd.Series(index=data.index, dtype=float)
    
    for i in range(5, len(data)):
        short_price_weighted.iloc[i] = sum(short_weights[j] * np.log(data['close'].iloc[i-j] / data['close'].iloc[i-j-5]) 
                                         for j in range(5))
        short_volume_weighted.iloc[i] = sum(short_weights[j] * (data['volume'].iloc[i-j] / data['volume'].iloc[i-j-5]) 
                                          for j in range(5))
    
    # Calculate medium-term momentum components (20-day)
    medium_price_momentum = np.log(data['close'] / data['close'].shift(20))
    medium_volume_momentum = data['volume'] / data['volume'].shift(20)
    
    # Apply exponential decay for medium-term (20-day window)
    medium_weights = np.exp(-0.1 * np.arange(20, 0, -1))
    medium_weights /= medium_weights.sum()
    
    # Calculate weighted medium-term momentum
    medium_price_weighted = pd.Series(index=data.index, dtype=float)
    medium_volume_weighted = pd.Series(index=data.index, dtype=float)
    
    for i in range(20, len(data)):
        medium_price_weighted.iloc[i] = sum(medium_weights[j] * np.log(data['close'].iloc[i-j] / data['close'].iloc[i-j-20]) 
                                          for j in range(20))
        medium_volume_weighted.iloc[i] = sum(medium_weights[j] * (data['volume'].iloc[i-j] / data['volume'].iloc[i-j-20]) 
                                           for j in range(20))
    
    # Calculate convergence score
    short_term_convergence = short_price_weighted * short_volume_weighted
    medium_term_convergence = medium_price_weighted * medium_volume_weighted
    
    # Weighted average convergence
    convergence_score = 0.7 * short_term_convergence + 0.3 * medium_term_convergence
    
    # Calculate 20-day price volatility
    log_returns = np.log(data['close'] / data['close'].shift(1))
    volatility = log_returns.rolling(window=20).std()
    
    # Volatility adjustment
    volatility_adjusted_convergence = convergence_score / volatility
    
    # Generate final factor values
    for i in range(20, len(data)):
        conv = volatility_adjusted_convergence.iloc[i]
        vol = volatility.iloc[i]
        
        if conv > 2 * vol:
            factor_values.iloc[i] = 2.0  # Strong buy
        elif conv > vol:
            factor_values.iloc[i] = 1.0  # Moderate buy
        elif abs(conv) <= vol:
            factor_values.iloc[i] = 0.0  # Neutral
        elif conv < -vol:
            factor_values.iloc[i] = -1.0  # Moderate sell
        elif conv < -2 * vol:
            factor_values.iloc[i] = -2.0  # Strong sell
        else:
            factor_values.iloc[i] = 0.0  # Neutral (fallback)
    
    return factor_values
