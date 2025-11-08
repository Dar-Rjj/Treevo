import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Volume-Adjusted Momentum factor
    Combines short, medium, and long-term momentum with volatility adjustment and volume confirmation
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize output series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # Calculate momentum components
    data['momentum_1d'] = data['close'] / data['close'].shift(1) - 1
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    
    # Blended momentum with timeframe diversification
    weights = [0.4, 0.35, 0.25]  # short, medium, long-term weights
    data['blended_momentum'] = (weights[0] * data['momentum_1d'] + 
                               weights[1] * data['momentum_3d'] + 
                               weights[2] * data['momentum_5d'])
    
    # Volatility adjustment - 5-day high-low range
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['volatility_5d'] = data['daily_range'].rolling(window=5, min_periods=3).mean()
    
    # Volatility-scaled momentum
    data['vol_scaled_momentum'] = data['blended_momentum'] / (data['volatility_5d'] + 1e-8)
    
    # Volume component
    data['baseline_volume'] = data['volume'].shift(1).rolling(window=5, min_periods=3).mean()
    data['volume_acceleration'] = data['volume'] / (data['baseline_volume'] + 1e-8)
    data['volume_momentum'] = data['volume'] / data['volume'].shift(1) - 1
    data['combined_volume'] = data['volume_acceleration'] * (1 + data['volume_momentum'])
    
    # Signal integration - volume-weighted adjusted momentum
    data['volume_weighted_factor'] = data['vol_scaled_momentum'] * data['combined_volume']
    
    # Cross-sectional ranking
    def cross_sectional_rank(group):
        return group.rank(pct=True)
    
    # Apply cross-sectional ranking by date
    alpha = data.groupby(data.index)['volume_weighted_factor'].transform(cross_sectional_rank)
    
    return alpha
