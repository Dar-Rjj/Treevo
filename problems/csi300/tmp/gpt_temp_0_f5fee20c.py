import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate typical price
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    
    # 1. Momentum-Adjusted Volume Divergence
    # Volume-to-Typical Price Ratio
    volume_typical_ratio = data['volume'] / typical_price
    
    # Deviation from 5-day average ratio
    volume_divergence = volume_typical_ratio - volume_typical_ratio.rolling(window=5, min_periods=3).mean()
    
    # 5-day price momentum
    momentum_5d = data['close'] / data['close'].shift(5) - 1
    
    # 10-day average true range
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10d = true_range.rolling(window=10, min_periods=5).mean()
    
    # Momentum-adjusted volume divergence
    factor1 = (volume_divergence * np.sign(momentum_5d)) / atr_10d
    
    # 2. Price Efficiency with Volume Surprise
    # Price inefficiency score
    price_range = data['high'] - data['low']
    price_range = price_range.replace(0, np.nan)  # Avoid division by zero
    inefficiency = 1 - (abs(data['close'] - data['close'].shift(1)) / price_range)
    
    # 3-day return sign
    ret_3d = data['close'] / data['close'].shift(3) - 1
    inefficiency_directed = inefficiency * np.sign(ret_3d)
    
    # Volume surprise vs 20-day average
    volume_surprise = data['volume'] / data['volume'].rolling(window=20, min_periods=10).mean() - 1
    
    # Price efficiency factor
    factor2 = inefficiency_directed * volume_surprise
    
    # 3. Spread-Pressure Interaction
    # Spread momentum
    spread_ratio = (data['high'] - data['low']) / typical_price
    spread_momentum = spread_ratio / spread_ratio.rolling(window=10, min_periods=5).mean() - 1
    
    # Weight by volume ratio to average
    volume_ratio = data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean()
    spread_weighted = spread_momentum * volume_ratio
    
    # Price pressure (close-to-open return magnitude)
    price_pressure = abs(data['close'] / data['open'] - 1)
    
    # Spread-pressure factor
    factor3 = spread_weighted * price_pressure
    
    # 4. Volume-Weighted Acceleration
    # Price acceleration
    ret_3d = data['close'] / data['close'].shift(3) - 1
    ret_5d = data['close'] / data['close'].shift(5) - 1
    price_acceleration = ret_3d - ret_5d
    
    # 10-day return volatility
    daily_returns = data['close'].pct_change()
    vol_10d = daily_returns.rolling(window=10, min_periods=5).std()
    
    # Normalized acceleration
    acceleration_norm = price_acceleration / vol_10d
    
    # Volume percentile (20-day window)
    volume_percentile = data['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # 5-day volume momentum
    volume_momentum = data['volume'] / data['volume'].shift(5) - 1
    
    # Dynamic volume weight
    volume_weight = volume_percentile * (1 + volume_momentum)
    
    # Volume-weighted acceleration
    factor4 = acceleration_norm * volume_weight
    
    # Combine all factors with equal weights
    combined_factor = (
        factor1.fillna(0) * 0.25 + 
        factor2.fillna(0) * 0.25 + 
        factor3.fillna(0) * 0.25 + 
        factor4.fillna(0) * 0.25
    )
    
    # Final normalization
    final_factor = (combined_factor - combined_factor.rolling(window=20, min_periods=10).mean()) / combined_factor.rolling(window=20, min_periods=10).std()
    
    return final_factor
