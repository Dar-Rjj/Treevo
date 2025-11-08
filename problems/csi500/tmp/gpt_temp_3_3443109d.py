import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Normalized Momentum with Volume Confirmation
    momentum_10d = data['close'] / data['close'].shift(10) - 1
    
    # Calculate rolling volatility using high-low range
    daily_range = data['high'] - data['low']
    volatility_20d = daily_range.rolling(window=20).std()
    
    # Volume ratio
    volume_ratio = data['volume'] / data['volume'].rolling(window=20).mean()
    
    # Combine components
    factor1 = momentum_10d / (volatility_20d + 1e-8) * volume_ratio
    
    # Price-Range Efficiency Factor
    prev_close = data['close'].shift(1)
    true_range = pd.DataFrame({
        'hl': data['high'] - data['low'],
        'hc': abs(data['high'] - prev_close),
        'lc': abs(data['low'] - prev_close)
    }).max(axis=1)
    
    avg_true_range_10d = true_range.rolling(window=10).mean()
    net_movement_10d = data['close'] - data['close'].shift(10)
    cumulative_true_range = true_range.rolling(window=10).sum()
    
    efficiency_ratio = net_movement_10d / (cumulative_true_range + 1e-8)
    efficiency_ratio = efficiency_ratio * np.sign(net_movement_10d)
    factor2 = efficiency_ratio
    
    # Volume-Weighted Intraday Pressure
    mid_point = (data['high'] + data['low']) / 2
    intraday_pressure = (data['close'] - mid_point) / (data['high'] - data['low'] + 1e-8)
    
    # Volume percentile rank
    volume_rank = data['volume'].rolling(window=15, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    pressure_volume = intraday_pressure * volume_rank
    factor3 = pressure_volume.rolling(window=5).mean()
    
    # Multi-Timeframe Trend Consistency
    short_term_dir = np.sign(data['close'] - data['close'].shift(5))
    medium_term_dir = np.sign(data['close'] - data['close'].shift(15))
    long_term_dir = np.sign(data['close'] - data['close'].shift(30))
    
    # Count consistent directions
    direction_count = (short_term_dir == medium_term_dir).astype(int) + \
                     (short_term_dir == long_term_dir).astype(int) + \
                     (medium_term_dir == long_term_dir).astype(int)
    
    # Weight by magnitude
    short_term_mag = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    medium_term_mag = (data['close'] - data['close'].shift(15)) / data['close'].shift(15)
    long_term_mag = (data['close'] - data['close'].shift(30)) / data['close'].shift(30)
    
    avg_magnitude = (abs(short_term_mag) + abs(medium_term_mag) + abs(long_term_mag)) / 3
    factor4 = direction_count * avg_magnitude
    
    # Liquidity-Adjusted Return Momentum
    momentum_20d = data['close'] / data['close'].shift(20) - 1
    
    # Liquidity proxy: dollar volume per unit price movement
    price_range = data['high'] - data['low']
    dollar_volume = data['amount']
    liquidity_proxy = dollar_volume / (price_range + 1e-8)
    
    # Apply exponential weighting
    liquidity_adjusted = momentum_20d * liquidity_proxy
    factor5 = liquidity_adjusted.ewm(span=10).mean()
    
    # Volatility-Regime Adaptive Factor
    current_volatility = daily_range.rolling(window=20).std()
    historical_volatility = daily_range.rolling(window=60).std()
    vol_ratio = current_volatility / historical_volatility
    
    # Regime-specific momentum
    momentum_5d = data['close'] / data['close'].shift(5) - 1
    momentum_15d = data['close'] / data['close'].shift(15) - 1
    
    regime_momentum = np.where(vol_ratio > 1, momentum_5d, momentum_15d)
    
    # Volume confirmation and regime strength scaling
    vol_strength = abs(vol_ratio - 1)
    factor6 = regime_momentum * volume_ratio * vol_strength
    
    # Combine all factors with equal weights
    combined_factor = (factor1.fillna(0) + factor2.fillna(0) + factor3.fillna(0) + 
                      factor4.fillna(0) + factor5.fillna(0) + factor6.fillna(0)) / 6
    
    return combined_factor
