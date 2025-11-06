import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-timeframe Momentum Acceleration
    ultra_short_momentum = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    short_term_momentum = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    medium_term_momentum = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    
    acceleration_score = (ultra_short_momentum - short_term_momentum) + (short_term_momentum - medium_term_momentum)
    
    def count_sign_persistence(series, window):
        sign_series = np.sign(series)
        persistence = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            window_data = sign_series.iloc[i-window+1:i+1]
            persistence.iloc[i] = (window_data == window_data.iloc[-1]).sum()
        return persistence
    
    acceleration_persistence = count_sign_persistence(acceleration_score, 3)
    
    # Volatility-scaled Breakout Detection
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=20).mean()
    
    high_breakout_intensity = (df['high'] - df['high'].rolling(window=20).max().shift(1)) / atr
    low_breakdown_intensity = (df['low'].rolling(window=20).min().shift(1) - df['low']) / atr
    net_breakout_signal = high_breakout_intensity - low_breakdown_intensity
    
    # Volume Confirmation System
    volume_momentum = (df['volume'] - df['volume'].shift(5)) / df['volume'].shift(5)
    volume_spike_ratio = df['volume'] / df['volume'].rolling(window=20).mean()
    volume_price_alignment = np.sign(volume_momentum) * np.sign(ultra_short_momentum)
    volume_confirmation_strength = volume_spike_ratio * volume_price_alignment
    
    volume_regime_multiplier = np.where(volume_spike_ratio > 1.5, 1.3, 
                                       np.where(volume_spike_ratio < 0.7, 0.7, 1.0))
    
    # Signal Integration
    core_momentum_signal = acceleration_score * acceleration_persistence
    breakout_enhanced = core_momentum_signal * net_breakout_signal
    volume_confirmed = breakout_enhanced * volume_confirmation_strength
    regime_adjusted = volume_confirmed * volume_regime_multiplier
    
    # Alpha Factor Construction
    signal_consistency = count_sign_persistence(regime_adjusted, 5)
    signal_magnitude = abs(regime_adjusted)
    quality_score = signal_consistency * signal_magnitude
    
    final_alpha_factor = regime_adjusted * quality_score
    
    return final_alpha_factor
