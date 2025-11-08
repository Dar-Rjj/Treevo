import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Short-Term Amplitude Momentum
    short_return = df['close'] / df['close'].shift(5) - 1
    
    # Calculate average high-low range for short-term (t-5 to t)
    short_range = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 5:
            short_range.iloc[i] = (df['high'].iloc[i-5:i+1] - df['low'].iloc[i-5:i+1]).mean()
    
    short_amplitude_momentum = short_return / short_range
    
    # Calculate Medium-Term Amplitude Momentum
    medium_return = df['close'] / df['close'].shift(20) - 1
    
    # Calculate average high-low range for medium-term (t-20 to t)
    medium_range = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 20:
            medium_range.iloc[i] = (df['high'].iloc[i-20:i+1] - df['low'].iloc[i-20:i+1]).mean()
    
    medium_amplitude_momentum = medium_return / medium_range
    
    # Calculate Divergence Signal
    divergence_signal = np.abs(short_amplitude_momentum - medium_amplitude_momentum)
    
    # Apply Volume-Weighted Efficiency Filter
    volume_weighted_divergence = df['volume'] * divergence_signal
    
    # Calculate Intraday Trading Efficiency
    current_range = df['high'] - df['low']
    trading_efficiency = df['amount'] / current_range
    
    # Combine with Volume-Weighted Divergence
    amplified_divergence = volume_weighted_divergence * trading_efficiency
    
    # Add Volatility Asymmetry Confirmation
    # Calculate Volatility Regime Change
    current_vol_intensity = (df['high'] - df['low']) / df['close']
    prev_vol_intensity = (df['high'].shift(1) - df['low'].shift(1)) / df['close'].shift(1)
    volatility_change_ratio = current_vol_intensity / prev_vol_intensity - 1
    
    # Determine Opening Gap Direction
    gap_direction = np.sign(df['open'] / df['close'].shift(1) - 1)
    
    # Multiply Final Factor Components
    final_factor = amplified_divergence * volatility_change_ratio * gap_direction
    
    return final_factor
