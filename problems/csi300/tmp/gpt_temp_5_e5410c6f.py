import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Momentum Components
    # Price Momentum
    data['price_momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['price_momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    
    # Volume Momentum
    data['volume_momentum_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_momentum_10d'] = data['volume'].shift(5) / data['volume'].shift(10) - 1
    
    # Apply Momentum Decay
    # Calculate Recent Volatility (10-day standard deviation of returns)
    data['daily_returns'] = data['close'].pct_change()
    data['volatility_10d'] = data['daily_returns'].rolling(window=10).std()
    
    # Apply Exponential Weighting based on volatility
    decay_rate = 1 / (1 + data['volatility_10d'])
    data['decayed_momentum'] = (
        data['price_momentum_5d'] * decay_rate + 
        data['price_momentum_10d'] * (1 - decay_rate)
    )
    
    # Detect Reversal Patterns
    # Identify Local Extrema
    data['is_local_min'] = (
        (data['close'] < data['close'].shift(1)) & 
        (data['close'] < data['close'].shift(-1)) & 
        (data['volume'] > 1.5 * data['volume'].shift(1))
    )
    data['is_local_max'] = (
        (data['close'] > data['close'].shift(1)) & 
        (data['close'] > data['close'].shift(-1)) & 
        (data['volume'] > 1.5 * data['volume'].shift(1))
    )
    
    # Calculate Reversal Strength
    data['reversal_strength'] = 0
    # For local minima (potential bullish reversal)
    min_mask = data['is_local_min']
    data.loc[min_mask, 'reversal_strength'] = (
        (data['close'].shift(1) - data['close']) / data['close'].shift(1)
    )
    # For local maxima (potential bearish reversal)
    max_mask = data['is_local_max']
    data.loc[max_mask, 'reversal_strength'] = (
        (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    )
    
    # Volume Confirmation Ratio
    data['volume_confirmation'] = data['volume'] / data['volume'].shift(1)
    
    # Volume-Scaling Mechanism
    # Relative Volume Position
    data['volume_20d_avg'] = data['volume'].rolling(window=20).mean()
    data['volume_ratio'] = data['volume'] / data['volume_20d_avg']
    
    # Calculate Volume Percentile Rank (20-day window)
    data['volume_percentile'] = data['volume'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] > x).mean(), raw=False
    )
    
    # Apply Volume Scaling to reversal strength
    data['volume_scaled_reversal'] = (
        data['reversal_strength'] * 
        np.where(data['volume_ratio'] > 1, data['volume_ratio'], 0.5)
    )
    
    # Volume-momentum alignment check
    data['momentum_volume_alignment'] = (
        np.sign(data['price_momentum_5d']) * np.sign(data['volume_momentum_5d'])
    )
    
    # Generate Composite Signal
    # Combine Decayed Momentum with Volume-Scaled Reversal
    data['composite_signal'] = (
        data['decayed_momentum'] * 0.6 + 
        data['volume_scaled_reversal'] * 0.4
    )
    
    # Apply momentum confirmation filter
    momentum_confirmation = np.sign(data['decayed_momentum']) == np.sign(data['volume_scaled_reversal'])
    data['composite_signal'] = data['composite_signal'] * np.where(momentum_confirmation, 1.5, 0.7)
    
    # Signal Strength Assessment
    conditions = [
        (momentum_confirmation & (data['volume_percentile'] > 0.7)),  # Strong
        ((~momentum_confirmation) & (data['volume_percentile'] > 0.5)),  # Moderate
        (data['volume_percentile'] <= 0.3)  # Weak
    ]
    choices = [1.2, 1.0, 0.5]
    data['signal_strength_multiplier'] = np.select(conditions, choices, default=1.0)
    
    # Final factor value
    data['factor'] = data['composite_signal'] * data['signal_strength_multiplier']
    
    return data['factor']
