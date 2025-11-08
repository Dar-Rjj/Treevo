import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Momentum Component
    # Multi-Timeframe Returns
    returns_5d = df['close'] / df['close'].shift(5) - 1
    returns_10d = df['close'] / df['close'].shift(10) - 1
    returns_21d = df['close'] / df['close'].shift(21) - 1
    
    # Volatility-Adjusted Momentum
    daily_returns = df['close'].pct_change()
    rolling_vol = daily_returns.rolling(window=10).std()
    vol_adjusted_5d = returns_5d / (rolling_vol + 0.0001)
    vol_adjusted_10d = returns_10d / (rolling_vol + 0.0001)
    vol_adjusted_21d = returns_21d / (rolling_vol + 0.0001)
    
    # Volume Confirmation Component
    volume_percentile = df['volume'].rolling(window=20).apply(
        lambda x: (x.rank(pct=True, method='average').iloc[-1]) * 100
    )
    volume_confidence = (volume_percentile / 100) * (1 - volume_percentile / 100)
    
    # Regime Detection
    # Market State Indicator
    market_state = np.where(returns_5d > returns_21d, 1.0, 0.5)
    
    # Volatility Regime
    vol_median = rolling_vol.rolling(window=20).median()
    volatility_regime = np.where(rolling_vol > vol_median, 0.8, 1.0)
    
    # Factor Combination
    # Momentum Aggregation (Weighted geometric mean)
    momentum_agg = (
        np.sign(vol_adjusted_5d) * 
        (np.abs(vol_adjusted_5d) ** 0.4) * 
        (np.abs(vol_adjusted_10d) ** 0.35) * 
        (np.abs(vol_adjusted_21d) ** 0.25)
    )
    
    # Volume Integration
    momentum_with_volume = momentum_agg * volume_confidence
    
    # Regime Adjustment
    final_alpha = momentum_with_volume * market_state * volatility_regime
    
    return pd.Series(final_alpha, index=df.index)
