import pandas as pd

def heuristics_v2(df):
    # Calculate the historical average of the close price over a 50-day window
    historical_avg = df['close'].rolling(window=50).mean()
    
    # Momentum factor: current close price divided by the historical average
    momentum_factor = df['close'] / historical_avg
    
    # Recent volatility: standard deviation of the log returns over a 20-day window
    log_returns = np.log(df['close'] / df['close'].shift(1))
    recent_volatility = log_returns.rolling(window=20).std()
    
    # Volatility adjustment: if the recent volatility is high, we slightly decrease the momentum factor, and vice versa
    volatility_adjustment = 1 - (recent_volatility - recent_volatility.mean()) / recent_volatility.std()
    adjusted_momentum = momentum_factor * volatility_adjustment
    
    # Output as a Series with the same index as input
    heuristics_matrix = pd.Series(adjusted_momentum, name='HeuristicFactor')
    
    return heuristics_matrix
