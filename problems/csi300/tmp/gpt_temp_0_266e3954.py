import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Multi-Timeframe Momentum
    short_momentum = df['close'] / df['close'].shift(5) - 1
    medium_momentum = df['close'] / df['close'].shift(20) - 1
    momentum_ratio = short_momentum / (medium_momentum + 1e-8)
    
    # Assess Volatility Regime
    daily_range = df['high'] - df['low']
    range_volatility = daily_range.rolling(window=20).mean()
    
    daily_returns = df['close'].pct_change()
    return_volatility = daily_returns.rolling(window=20).std()
    
    combined_volatility = np.maximum(range_volatility, return_volatility)
    
    # Detect Volume-Price Divergence
    volume_momentum = df['volume'] / df['volume'].shift(5) - 1
    
    # Calculate 5-day rolling correlation between returns and volume
    volume_price_corr = pd.Series(index=df.index, dtype=float)
    for i in range(5, len(df)):
        window_returns = daily_returns.iloc[i-4:i+1]
        window_volume = df['volume'].iloc[i-4:i+1]
        if len(window_returns) >= 2 and len(window_volume) >= 2:
            volume_price_corr.iloc[i] = window_returns.corr(window_volume)
        else:
            volume_price_corr.iloc[i] = 0
    
    # Generate Composite Alpha Factor
    volatility_adjusted_momentum = momentum_ratio / (combined_volatility + 1e-8)
    alpha_factor = volatility_adjusted_momentum * volume_price_corr
    
    return alpha_factor
