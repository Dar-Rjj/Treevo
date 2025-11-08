import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate raw intraday reversal signal
    high_low_range = df['high'] - df['low']
    high_low_range = high_low_range.replace(0, np.nan)  # Avoid division by zero
    
    raw_reversal = ((df['high'] - df['close']) / high_low_range) - ((df['close'] - df['low']) / high_low_range)
    
    # Calculate 10-day historical volatility
    close_returns = df['close'].pct_change()
    historical_vol = close_returns.rolling(window=10, min_periods=1).std() / df['close'].rolling(window=10, min_periods=1).mean()
    historical_vol = historical_vol.replace(0, np.nan)  # Avoid division by zero
    
    # Incorporate volatility regime adjustment
    volatility_adjusted_reversal = raw_reversal / historical_vol
    
    # Calculate volume percentile (20-day lookback)
    volume_percentile = df['volume'].rolling(window=20, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Identify high volume clusters
    volume_cluster = (volume_percentile > 0.8) & (volume_percentile.shift(1) > 0.8)
    
    # Apply volume clustering filter
    volume_multiplier = np.where(volume_cluster, 1.0, 0.5)
    final_factor = volatility_adjusted_reversal * volume_multiplier
    
    return final_factor
