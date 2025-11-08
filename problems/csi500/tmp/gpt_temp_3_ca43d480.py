import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Volume-Weighted Momentum Factor
    Combines short-term and medium-term momentum with volume dynamics,
    volatility adjustment, and price gap signals
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Short-term Momentum Component (1-day)
    # Price return
    price_return_1d = data['close'] / data['close'].shift(1) - 1
    
    # Volume acceleration
    volume_accel = data['volume'] / data['volume'].shift(1)
    
    # Medium-term Momentum Component (3-day)
    # Price return
    price_return_3d = data['close'] / data['close'].shift(3) - 1
    
    # Volume trend (slope over 3 days)
    volume_trend = pd.Series(np.zeros(len(data)), index=data.index)
    for i in range(2, len(data)):
        if i >= 2:
            x = np.array([0, 1, 2])
            y = data['volume'].iloc[i-2:i+1].values
            if len(y) == 3 and not np.any(np.isnan(y)):
                slope = np.polyfit(x, y, 1)[0]
                volume_trend.iloc[i] = slope / np.mean(y) if np.mean(y) != 0 else 0
    
    # Volatility Adjustment
    # Price range based volatility proxy
    daily_range = (data['high'] - data['low']) / data['close']
    vol_adjustment = daily_range.rolling(window=5, min_periods=3).mean()
    
    # Price Gap Component
    overnight_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Factor Combination
    # Blend momentum components with volume weights
    short_term_momentum = price_return_1d * np.log1p(np.abs(volume_accel - 1))
    medium_term_momentum = price_return_3d * np.log1p(np.abs(volume_trend))
    
    # Combine momentum components (60% short-term, 40% medium-term)
    combined_momentum = 0.6 * short_term_momentum + 0.4 * medium_term_momentum
    
    # Adjust by volatility (inverse relationship)
    volatility_adjusted = combined_momentum / (vol_adjustment + 1e-6)
    
    # Incorporate price gap signals
    gap_adjusted = volatility_adjusted * (1 + np.sign(overnight_gap) * np.abs(overnight_gap))
    
    # Final factor with cross-sectional ranking
    factor = gap_adjusted.rank(pct=True)
    
    return factor
