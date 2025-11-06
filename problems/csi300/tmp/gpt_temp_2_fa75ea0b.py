import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Position Momentum
    # Short-term position (4-day lookback)
    short_position = (data['close'] - data['low'].shift(4)) / (data['high'].shift(4) - data['low'].shift(4))
    
    # Medium-term position (9-day lookback)
    medium_position = (data['close'] - data['low'].shift(9)) / (data['high'].shift(9) - data['low'].shift(9))
    
    # Long-term position (19-day lookback)
    long_position = (data['close'] - data['low'].shift(19)) / (data['high'].shift(19) - data['low'].shift(19))
    
    # Acceleration Signals
    short_medium_accel = medium_position - short_position
    medium_long_accel = long_position - medium_position
    
    # Volume-Price Efficiency
    # Price Efficiency (current day)
    price_efficiency = (data['close'] - data['low']) / (data['high'] - data['low'])
    price_efficiency = price_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # True Range calculation
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Volume Confirmation
    volume_confirmation = data['volume'] / true_range
    volume_confirmation = volume_confirmation.replace([np.inf, -np.inf], np.nan)
    
    # Amount-Based Pressure
    # Trade Size (avoid division by zero)
    trade_size = data['amount'] / data['volume']
    trade_size = trade_size.replace([np.inf, -np.inf], np.nan)
    
    # Price Impact
    price_impact = data['close'] - data['open']
    
    # Composite Signal
    # Momentum-Volume component using both acceleration signals
    momentum_volume_short = short_medium_accel * volume_confirmation
    momentum_volume_long = medium_long_accel * volume_confirmation
    
    # Combined momentum-volume signal
    combined_momentum_volume = 0.6 * momentum_volume_short + 0.4 * momentum_volume_long
    
    # Trade Size Weighting
    # Normalize trade size for weighting
    trade_size_rank = trade_size.rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() != 0 else 0
    )
    
    # Final composite factor
    composite_factor = combined_momentum_volume * trade_size_rank * price_impact
    
    # Handle any remaining infinite values
    composite_factor = composite_factor.replace([np.inf, -np.inf], np.nan)
    
    return composite_factor
