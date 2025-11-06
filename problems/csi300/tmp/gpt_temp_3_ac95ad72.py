import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Divergence with Intraday Momentum Confirmation factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Compute Price-Volume Divergence
    # Calculate 10-day price returns and volume changes
    data['price_return_10d'] = data['close'].pct_change(periods=10)
    data['volume_change_10d'] = data['volume'].pct_change(periods=10)
    
    # Calculate rolling correlation between price returns and volume changes
    data['price_volume_corr'] = data['price_return_10d'].rolling(window=10).corr(data['volume_change_10d'])
    
    # Identify divergence patterns
    data['bullish_divergence'] = ((data['price_return_10d'] < 0) & (data['volume_change_10d'] > 0)).astype(int)
    data['bearish_divergence'] = ((data['price_return_10d'] > 0) & (data['volume_change_10d'] < 0)).astype(int)
    
    # Calculate Intraday Momentum Strength
    data['midpoint'] = (data['high'] + data['low']) / 2
    data['momentum_direction'] = data['close'] - data['midpoint']
    data['momentum_strength'] = np.abs(data['momentum_direction']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volume Confirmation Analysis
    data['volume_ma_20'] = data['volume'].rolling(window=20).mean()
    data['abnormal_volume'] = data['volume'] / data['volume_ma_20']
    
    # Calculate volume persistence (consistency of volume spikes)
    data['volume_spike'] = (data['abnormal_volume'] > 1.2).astype(int)
    data['volume_persistence'] = data['volume_spike'].rolling(window=5).sum()
    
    # Volume trend alignment (slope of volume MA)
    data['volume_trend'] = data['volume_ma_20'].rolling(window=10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    
    # Generate Composite Signal
    # Divergence-Momentum Alignment
    data['bullish_alignment'] = ((data['price_volume_corr'] < 0) & (data['momentum_direction'] > 0)).astype(int)
    data['bearish_alignment'] = ((data['price_volume_corr'] > 0) & (data['momentum_direction'] < 0)).astype(int)
    
    # Strength Weighting
    data['momentum_multiplier'] = data['momentum_strength'] * 2  # Amplify momentum effect
    data['volume_multiplier'] = np.minimum(data['abnormal_volume'], 3)  # Cap volume effect
    
    # Signal Validation components
    data['price_reversal'] = ((data['close'] > data['open']) & (data['close'].shift(1) < data['open'].shift(1))).astype(int)
    data['volume_acceleration'] = (data['volume'] > data['volume'].shift(1)).astype(int)
    
    # Final composite factor calculation
    # Base signal from divergence-momentum alignment
    base_signal = np.where(
        data['bullish_alignment'] == 1, 1,
        np.where(data['bearish_alignment'] == 1, -1, 0)
    )
    
    # Apply strength weighting
    weighted_signal = base_signal * data['momentum_multiplier'] * data['volume_multiplier']
    
    # Signal validation filtering
    validated_signal = np.where(
        (data['volume_persistence'] >= 2) & (data['volume_trend'] > 0),
        weighted_signal,
        weighted_signal * 0.5  # Reduce signal strength if volume conditions not met
    )
    
    # Final factor with additional validation
    final_factor = np.where(
        (data['price_reversal'] == 1) | (data['volume_acceleration'] == 1),
        validated_signal,
        validated_signal * 0.7  # Further reduce if no confirmation signals
    )
    
    # Return as pandas Series with proper index
    return pd.Series(final_factor, index=data.index, name='price_volume_divergence_momentum')
