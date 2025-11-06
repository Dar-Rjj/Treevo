import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Regime-Shifted Momentum Acceleration with Volume-Price Divergence alpha factor
    """
    # Calculate Momentum Acceleration Components
    # Short-term momentum (5-day)
    price_return_5 = df['close'] / df['close'].shift(5) - 1
    volume_return_5 = df['volume'] / df['volume'].shift(5) - 1
    
    # Medium-term momentum (10-day)
    price_return_10 = df['close'] / df['close'].shift(10) - 1
    volume_return_10 = df['volume'] / df['volume'].shift(10) - 1
    
    # Momentum acceleration
    price_acceleration = price_return_10 - price_return_5
    volume_acceleration = volume_return_10 - volume_return_5
    
    # Volume-Price Divergence Patterns
    divergence_score = price_acceleration * (-volume_acceleration)
    
    # Determine Market Regime Context
    # Volatility regime - 10-day price range
    high_10 = df['high'].rolling(window=10).max()
    low_10 = df['low'].rolling(window=10).min()
    price_range_10 = (high_10 - low_10) / df['close'].shift(1)
    avg_range_20 = price_range_10.rolling(window=20).mean()
    volatility_regime = price_range_10 / avg_range_20
    
    # Trend regime - 20-day price slope
    def calc_slope(x):
        if len(x) < 2:
            return np.nan
        return stats.linregress(range(len(x)), x)[0]
    
    trend_slope = df['close'].rolling(window=20).apply(calc_slope, raw=True)
    trend_regime = np.sign(trend_slope) * np.abs(trend_slope) / df['close'].shift(1)
    
    # Regime-Based Signal Adjustment
    # Amplify in high volatility, adjust by trend direction
    regime_weight = volatility_regime * (1 + np.abs(trend_regime))
    
    # Generate Composite Alpha Factor
    alpha_signal = divergence_score * price_acceleration * regime_weight
    
    return alpha_signal
