import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from arch import arch_model

def heuristics_v2(df, N=20, M=20):
    """
    Calculate a novel alpha factor: Volume Adjusted Momentum with Price Change Volatility.
    
    Parameters:
    - df: pandas DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume']
          indexed by (date).
    - N: int, number of days to compute the momentum.
    - M: int, number of days to compute the volatility.
    
    Returns:
    - pandas Series indexed by (date) representing the factor values.
    """
    
    # Obtain Close Prices
    close_prices = df['close']
    
    # Compute Log Return over N Days
    log_returns = np.log(close_prices).diff()
    
    # Calculate Momentum
    momentum = log_returns.rolling(window=N).sum()
    
    # Adjust for Volume
    volumes = df['volume']
    volume_relative = volumes / volumes.rolling(window=M).mean()
    volume_adjusted_momentum = momentum * volume_relative
    
    # Determine Absolute Price Changes
    abs_price_changes = close_prices.diff().abs()
    
    # Calculate Advanced Volatility Measure
    # Standard Deviation of Absolute Price Changes over M Days
    std_volatility = abs_price_changes.rolling(window=M).std()
    
    # Adaptive Volatility (e.g., GARCH)
    garch_volatility = []
    for i in range(len(log_returns)):
        if i < M:
            garch_volatility.append(np.nan)
        else:
            model = arch_model(log_returns.iloc[:i], vol='Garch', p=1, q=1, dist='Normal')
            res = model.fit(disp='off')
            garch_volatility.append(res.conditional_volatility[-1])
    garch_volatility = pd.Series(garch_volatility, index=log_returns.index)
    
    # Final Factor Calculation
    weighted_momentum = 0.6 * volume_adjusted_momentum
    weighted_volatility = 0.4 * garch_volatility
    final_factor = weighted_momentum + weighted_volatility
    
    return final_factor
