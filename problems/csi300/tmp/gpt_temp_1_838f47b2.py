import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Acceleration Component
    # Price momentum: (Close(t) - Close(t-5)) / Close(t-5)
    price_momentum = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    
    # Acceleration: (Close(t) - Close(t-5)) - (Close(t-5) - Close(t-10))
    acceleration = (data['close'] - data['close'].shift(5)) - (data['close'].shift(5) - data['close'].shift(10))
    
    # Momentum persistence: Count of positive returns in last 5 days
    returns = data['close'].pct_change()
    positive_returns_count = pd.Series(np.zeros(len(data)), index=data.index)
    for i in range(5, len(data)):
        positive_returns_count.iloc[i] = (returns.iloc[i-4:i+1] > 0).sum()
    
    # Normalize momentum components
    momentum_acceleration = (price_momentum.rank() + acceleration.rank() + positive_returns_count.rank()) / 3
    
    # Volume Divergence Signal
    # Volume trend: Volume(t) / 5-day average Volume
    volume_ma_5 = data['volume'].rolling(window=5, min_periods=1).mean()
    volume_trend = data['volume'] / volume_ma_5
    
    # Price-volume correlation: Sign(Close(t)-Close(t-1)) × Sign(Volume(t)-Volume(t-1))
    price_change_sign = np.sign(data['close'] - data['close'].shift(1))
    volume_change_sign = np.sign(data['volume'] - data['volume'].shift(1))
    price_volume_corr = price_change_sign * volume_change_sign
    
    # Divergence strength: ABS(Price momentum - Volume trend)
    divergence_strength = np.abs(price_momentum - volume_trend)
    
    # Combine volume divergence components
    volume_divergence = (volume_trend.rank() + price_volume_corr.rank() + divergence_strength.rank()) / 3
    
    # Market Regime Filter
    # Volatility regime: 10-day standard deviation of returns
    volatility_regime = returns.rolling(window=10, min_periods=1).std()
    
    # Trend regime: 20-day moving average slope
    ma_20 = data['close'].rolling(window=20, min_periods=1).mean()
    trend_regime = ma_20.diff(5) / ma_20.shift(5)
    
    # Regime score: Volatility regime × Trend regime
    regime_score = volatility_regime * trend_regime
    
    # Final factor: Combine momentum acceleration and volume divergence, weighted by regime filter
    factor = momentum_acceleration * volume_divergence * (1 + regime_score.rank())
    
    return factor
