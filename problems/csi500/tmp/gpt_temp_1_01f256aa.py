import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining intraday volatility-adjusted momentum and volume-synchronized price probability.
    """
    # Intraday Volatility-Adjusted Price Momentum
    # Calculate intraday price momentum using high and low prices
    intraday_range = (df['high'] - df['low']) / df['close']
    
    # Calculate recent price trend using moving averages
    short_ma = df['close'].rolling(window=5).mean()
    long_ma = df['close'].rolling(window=20).mean()
    price_trend = (short_ma - long_ma) / df['close']
    
    # Calculate volatility regime
    returns = df['close'].pct_change()
    vol_20d = returns.rolling(window=20).std()
    vol_regime = np.where(vol_20d > vol_20d.rolling(window=60).median(), 1, 0)  # 1 for high vol, 0 for low vol
    
    # Scale momentum based on volatility regime (amplify in low vol, dampen in high vol)
    volatility_adjusted_momentum = price_trend * (1 - 0.5 * vol_regime)
    
    # Volume-Synchronized Price Probability
    # Calculate volume-weighted price percentiles
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    volume_weighted_price = (typical_price * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # Calculate current price's percentile rank relative to recent distribution
    price_percentile = df['close'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0.5
    )
    
    # Extreme positions as reversal signals (U-shaped transformation)
    volume_sync_prob = -np.abs(price_percentile - 0.5) * 2 + 1
    
    # Combine both factors
    alpha_factor = volatility_adjusted_momentum * volume_sync_prob
    
    return alpha_factor
