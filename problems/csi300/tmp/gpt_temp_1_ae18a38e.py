import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Parameters
    N = 20  # Momentum lookback period
    M = 10  # Turnover ranking period
    K = 10  # Amihud ratio smoothing period
    
    # Momentum Persistence Component
    # Calculate daily returns
    daily_returns = data['close'].pct_change()
    
    # Compute N-day cumulative return
    cumulative_return = (1 + daily_returns).rolling(window=N).apply(np.prod, raw=True) - 1
    
    # Calculate rolling standard deviation of daily returns
    returns_volatility = daily_returns.rolling(window=N).std()
    
    # Risk-adjusted momentum (avoid division by zero)
    risk_adjusted_momentum = cumulative_return / (returns_volatility + 1e-8)
    
    # Liquidity Constraints Component
    # Trading Liquidity Assessment
    # Using volume as proxy for turnover rate (assuming shares outstanding is constant)
    daily_turnover = data['volume']
    
    # Calculate rolling percentile rank of turnover rate over M days
    def percentile_rank(x):
        if len(x) < 2:
            return np.nan
        return (x.rank(pct=True).iloc[-1])
    
    turnover_percentile = daily_turnover.rolling(window=M).apply(percentile_rank, raw=False)
    
    # Price Impact Analysis
    # Compute daily Amihud illiquidity ratio (absolute return / dollar volume)
    dollar_volume = data['volume'] * data['close']
    amihud_ratio = np.abs(daily_returns) / (dollar_volume + 1e-8)
    
    # Calculate rolling mean of Amihud ratio over K days
    amihud_mean = amihud_ratio.rolling(window=K).mean()
    
    # Factor Integration
    # Multiply momentum component by liquidity percentile rank
    momentum_liquidity = risk_adjusted_momentum * turnover_percentile
    
    # Invert and multiply by Amihud ratio mean for liquidity adjustment
    # Adding small constant to avoid division by zero
    factor = momentum_liquidity / (amihud_mean + 1e-8)
    
    return factor
