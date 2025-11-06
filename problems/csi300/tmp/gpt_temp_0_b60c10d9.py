import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Compressed Intraday Momentum Divergence
    # Calculate Intraday Momentum Components
    daily_range = data['high'] - data['low']
    close_position = (data['close'] - data['low']) / (daily_range.replace(0, np.nan))
    
    # 3-day momentum of close position
    close_pos_momentum = close_position - close_position.shift(3)
    
    # Calculate Short-Term Volatility (5-day ATR)
    tr = pd.DataFrame({
        'hl': data['high'] - data['low'],
        'hc': abs(data['high'] - data['close'].shift(1)),
        'lc': abs(data['low'] - data['close'].shift(1))
    }).max(axis=1)
    short_term_vol = tr.rolling(window=5, min_periods=3).mean()
    
    # Calculate Medium-Term Volatility (20-day ATR)
    medium_term_vol = tr.rolling(window=20, min_periods=10).mean()
    
    # Compute Volatility Compression Ratio
    volatility_ratio = short_term_vol / medium_term_vol.replace(0, np.nan)
    momentum_factor = close_pos_momentum * volatility_ratio
    
    # Volume-Stabilized Gap Mean Reversion
    # Calculate Gap Characteristics
    overnight_gap = data['open'] / data['close'].shift(1) - 1
    abs_gap = abs(overnight_gap)
    
    # Apply Volume Stabilization
    volume_std = data['volume'].rolling(window=10, min_periods=5).std()
    mean_reversion_factor = -overnight_gap * (1 / volume_std.replace(0, np.nan))
    
    # Amount-Weighted Range Breakout Efficiency
    # Measure Breakout Potential
    current_range = data['high'] - data['low']
    avg_range = current_range.rolling(window=5, min_periods=3).mean()
    compression_ratio = current_range / avg_range.replace(0, np.nan)
    
    # Apply Amount Weighting
    avg_amount = data['amount'].rolling(window=5, min_periods=3).mean()
    breakout_factor = (1 - compression_ratio) * (data['amount'] / avg_amount.replace(0, np.nan))
    
    # Price-Velocity Volume Profile Convergence
    # Compute Velocity Components
    velocity_3d = data['close'] - data['close'].shift(3)
    velocity_6d = data['close'] - data['close'].shift(6)
    velocity_ratio = velocity_3d / velocity_6d.replace(0, np.nan)
    
    # Apply Volume Profile Adjustment
    volume_percentile = data['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: (x[-1] > x[:-1]).mean() if len(x) == 20 else np.nan
    )
    velocity_factor = velocity_ratio * volume_percentile
    
    # Combine all factors with equal weighting
    combined_factor = (
        momentum_factor.fillna(0) + 
        mean_reversion_factor.fillna(0) + 
        breakout_factor.fillna(0) + 
        velocity_factor.fillna(0)
    )
    
    return combined_factor
