import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Acceleration Divergence Factor
    Combines price/volume acceleration, intraday pressure, and divergence patterns
    to predict short-term reversals
    """
    data = df.copy()
    
    # Calculate Acceleration Components
    # Price Acceleration
    price_3d_momentum = (data['close'].shift(1) - data['close'].shift(3)) / data['close'].shift(3)
    price_6d_momentum = (data['close'].shift(1) - data['close'].shift(6)) / data['close'].shift(6)
    price_acceleration = price_3d_momentum - price_6d_momentum
    
    # Volume Acceleration
    vol_3d_momentum = (data['volume'].shift(1) - data['volume'].shift(3)) / (data['volume'].shift(3) + 1e-8)
    vol_6d_momentum = (data['volume'].shift(1) - data['volume'].shift(6)) / (data['volume'].shift(6) + 1e-8)
    volume_acceleration = vol_3d_momentum - vol_6d_momentum
    
    # Acceleration Direction Alignment
    same_direction = (price_acceleration * volume_acceleration) > 0
    alignment_strength = 1 - np.abs(price_acceleration - volume_acceleration) / (np.abs(price_acceleration) + np.abs(volume_acceleration) + 1e-8)
    alignment_score = same_direction.astype(float) * alignment_strength
    
    # Calculate Intraday Pressure Components
    # Opening Gap Pressure
    gap_pressure_list = []
    efficiency_list = []
    
    for i in range(1, 6):  # t-5 to t-1
        prev_close = data['close'].shift(i+1)
        daily_open = data['open'].shift(i)
        daily_close = data['close'].shift(i)
        daily_high = data['high'].shift(i)
        daily_low = data['low'].shift(i)
        
        # Opening gap
        opening_gap = (daily_open / prev_close - 1)
        daily_return = (daily_close / daily_open - 1)
        
        # Gap pressure: gap size times return direction
        gap_pressure = opening_gap * np.sign(daily_return)
        
        # Intraday range efficiency
        daily_range = (daily_high - daily_low) / (daily_low + 1e-8)
        abs_return = np.abs(daily_close - daily_open) / daily_open
        efficiency = abs_return / (daily_range + 1e-8)
        
        gap_pressure_list.append(gap_pressure)
        efficiency_list.append(efficiency)
    
    # Pressure Accumulation
    gap_pressure_df = pd.DataFrame(gap_pressure_list).T
    efficiency_df = pd.DataFrame(efficiency_list).T
    
    # Weight gap pressure by efficiency
    weighted_pressure = gap_pressure_df * efficiency_df
    pressure_intensity = weighted_pressure.sum(axis=1) / (efficiency_df.sum(axis=1) + 1e-8)
    pressure_direction = np.sign(pressure_intensity)
    
    # Detect Divergence Patterns
    # Acceleration-Volume Divergence
    positive_divergence = (price_acceleration > 0) & (volume_acceleration < 0)
    negative_divergence = (price_acceleration < 0) & (volume_acceleration > 0)
    
    divergence_magnitude = np.where(
        positive_divergence,
        price_acceleration / (np.abs(volume_acceleration) + 1e-8),
        np.where(
            negative_divergence,
            price_acceleration / (np.abs(volume_acceleration) + 1e-8),
            0
        )
    )
    
    # Pressure-Acceleration Mismatch
    pressure_acc_mismatch = (pressure_direction * np.sign(price_acceleration)) < 0
    mismatch_severity = np.abs(pressure_intensity - price_acceleration)
    
    # Multi-timeframe Confirmation
    # Additional shorter timeframe acceleration
    price_2d_momentum = (data['close'].shift(1) - data['close'].shift(2)) / data['close'].shift(2)
    price_4d_momentum = (data['close'].shift(1) - data['close'].shift(4)) / data['close'].shift(4)
    short_term_acc = price_2d_momentum - price_4d_momentum
    
    # Check consistency across timeframes
    timeframe_consistency = (np.sign(price_acceleration) == np.sign(short_term_acc)).astype(float)
    
    # Pressure pattern consistency (last 3 days)
    recent_pressure_consistency = pressure_direction.rolling(window=3, min_periods=1).apply(
        lambda x: len(set(np.sign(x.dropna()))) == 1 if len(x.dropna()) > 0 else 0, raw=False
    )
    
    confirmation_strength = (timeframe_consistency + recent_pressure_consistency) / 2
    
    # Combine with Reversal Logic
    # Divergence-Reversal Mapping
    divergence_reversal_map = np.where(
        positive_divergence, 1,  # Bullish reversal potential
        np.where(negative_divergence, -1, 0)  # Bearish reversal potential
    )
    
    # Pressure Release Signal
    pressure_energy = np.abs(pressure_intensity) * (1 - np.abs(price_acceleration))
    pressure_release_prob = 1 / (1 + np.exp(-pressure_energy * 10))  # Sigmoid scaling
    
    # Integrated Factor Construction
    base_factor = divergence_magnitude * np.abs(pressure_intensity)
    directional_factor = base_factor * divergence_reversal_map
    confirmed_factor = directional_factor * confirmation_strength
    weighted_factor = confirmed_factor * alignment_score
    
    # Apply Dynamic Thresholding
    # Stock-Specific Volatility Adjustment
    recent_volatility = (data['high'].rolling(window=10, min_periods=5).max() - 
                        data['low'].rolling(window=10, min_periods=5).min()) / data['close'].rolling(window=10, min_periods=5).mean()
    
    # Volatility-based scaling
    vol_quantiles = recent_volatility.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).quantile(0.5) if len(x.dropna()) > 0 else 1, raw=False
    )
    
    volatility_adjustment = np.where(
        recent_volatility > vol_quantiles * 1.5, 0.7,  # High vol: reduce sensitivity
        np.where(recent_volatility < vol_quantiles * 0.5, 1.3, 1.0)  # Low vol: increase sensitivity
    )
    
    # Final Alpha Output
    final_factor = weighted_factor * volatility_adjustment * pressure_release_prob
    
    return pd.Series(final_factor, index=data.index, name='price_volume_acceleration_divergence')
