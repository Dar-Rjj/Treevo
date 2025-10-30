import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    High-Low Range Momentum Divergence factor combining range efficiency, 
    momentum divergence, and volume confirmation signals.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate True Range Efficiency
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    data['abs_price_change'] = abs(data['close'] - data['prev_close'])
    data['range_efficiency'] = np.where(
        data['true_range'] > 0,
        data['abs_price_change'] / data['true_range'],
        0
    )
    
    # Calculate Multi-period Price Momentum
    data['momentum_short'] = (data['close'] / data['close'].shift(5)) - 1
    data['momentum_medium'] = (data['close'] / data['close'].shift(20)) - 1
    
    # Calculate Volume Trend Momentum
    data['volume_short_slope'] = (
        data['volume'].rolling(window=5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan
        )
    )
    data['volume_medium_slope'] = (
        data['volume'].rolling(window=20).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else np.nan
        )
    )
    
    # Calculate momentum divergence
    data['momentum_divergence'] = data['momentum_short'] - data['momentum_medium']
    
    # Calculate volume confirmation
    data['volume_confirmation'] = np.where(
        (data['volume_short_slope'] > 0) & (data['volume_medium_slope'] > 0), 1,
        np.where(
            (data['volume_short_slope'] < 0) & (data['volume_medium_slope'] < 0), -1, 0
        )
    )
    
    # Combine Range Efficiency with Momentum Divergence
    data['range_momentum_signal'] = data['range_efficiency'] * data['momentum_divergence']
    
    # Apply Volume Confirmation
    data['factor'] = data['range_momentum_signal'] * (1 + 0.5 * data['volume_confirmation'])
    
    # Clean up intermediate columns
    result = data['factor'].copy()
    
    return result
