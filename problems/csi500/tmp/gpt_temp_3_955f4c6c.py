import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Scaled Multi-Timeframe Momentum with Volume Confirmation
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Pre-calculate basic components
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Daily price range
    daily_range = high - low
    
    # Volatility estimation (5-day average range)
    volatility_5d = daily_range.rolling(window=5, min_periods=3).mean()
    
    # Volatility regime detection
    avg_range_20d = daily_range.rolling(window=20, min_periods=10).mean()
    volatility_regime = pd.Series(index=df.index, dtype=str)
    volatility_regime[daily_range > 2 * avg_range_20d] = 'high'
    volatility_regime[daily_range < 0.5 * avg_range_20d] = 'low'
    volatility_regime = volatility_regime.fillna('normal')
    
    # Volume components
    volume_change = volume.pct_change()
    avg_volume_20d = volume.rolling(window=20, min_periods=10).mean()
    volume_above_avg = (volume > avg_volume_20d).astype(int)
    
    # Volume persistence (consecutive days above average)
    volume_persistence = volume_above_avg.copy()
    for i in range(1, len(volume_persistence)):
        if volume_above_avg.iloc[i] == 1:
            volume_persistence.iloc[i] = volume_persistence.iloc[i-1] + 1
        else:
            volume_persistence.iloc[i] = 0
    
    # Initialize signal storage
    short_term_signals = pd.Series(index=df.index, dtype=float)
    medium_term_signals = pd.Series(index=df.index, dtype=float)
    persistence_count = pd.Series(0, index=df.index, dtype=int)
    
    # Calculate signals for each day
    for i in range(1, len(df)):
        if i < 10:  # Need sufficient history
            continue
            
        current_date = df.index[i]
        
        # Short-term momentum (1-3 days)
        price_momentum = (close.iloc[i] - close.iloc[i-1]) / max(daily_range.iloc[i], 0.001)
        
        # Volume confirmation for short-term
        volume_acceleration = 1.0
        if volume.iloc[i] > volume.iloc[i-1]:
            volume_acceleration = 1.2 + min(volume_change.iloc[i], 0.5)
        
        short_term_signal = price_momentum * volume_acceleration
        
        # Medium-term momentum (5-10 days with decay)
        medium_returns = []
        weights = []
        decay_factor = 0.9
        
        for lookback in range(1, 11):
            if i - lookback >= 0:
                daily_return = (close.iloc[i] - close.iloc[i-lookback]) / close.iloc[i-lookback]
                weight = decay_factor ** lookback
                medium_returns.append(daily_return * weight)
                weights.append(weight)
        
        if weights:
            medium_term_signal = np.sum(medium_returns) / np.sum(weights)
        else:
            medium_term_signal = 0
        
        # Volume persistence check for medium-term
        volume_persistence_bonus = 1.0 + min(volume_persistence.iloc[i] * 0.05, 0.3)
        medium_term_signal *= volume_persistence_bonus
        
        # Multi-timeframe confirmation
        timeframe_alignment = 1.0
        if short_term_signal * medium_term_signal > 0:  # Same direction
            timeframe_alignment = 1.5
        elif short_term_signal * medium_term_signal < 0:  # Opposite direction
            timeframe_alignment = 0.5
        
        # Persistence tracking
        if i > 1:
            prev_short = short_term_signals.iloc[i-1]
            prev_medium = medium_term_signals.iloc[i-1]
            
            current_dir = np.sign(short_term_signal + medium_term_signal)
            prev_dir = np.sign(prev_short + prev_medium)
            
            if current_dir == prev_dir and current_dir != 0:
                persistence_count.iloc[i] = persistence_count.iloc[i-1] + 1
            else:
                persistence_count.iloc[i] = 1
        
        # Persistence weighting (capped at 5)
        persistence_weight = 1.0 + min(persistence_count.iloc[i] * 0.1, 0.5)
        
        # Store current signals
        short_term_signals.iloc[i] = short_term_signal
        medium_term_signals.iloc[i] = medium_term_signal
        
        # Combine signals with timeframe alignment
        combined_signal = (short_term_signal + medium_term_signal) * timeframe_alignment * persistence_weight
        
        # Volatility scaling
        if not pd.isna(volatility_5d.iloc[i]) and volatility_5d.iloc[i] > 0:
            volatility_scaling = 1.0 / volatility_5d.iloc[i]
            
            # Regime-specific adjustments
            regime = volatility_regime.iloc[i]
            if regime == 'high':
                volatility_scaling *= 0.7  # Reduce signal in high volatility
            elif regime == 'low':
                volatility_scaling *= 1.3  # Enhance signal in low volatility
            
            combined_signal *= volatility_scaling
        
        # Apply exponential decay to previous signals
        decay_factor = 0.85
        if i > 1 and not pd.isna(result.iloc[i-1]):
            decayed_previous = result.iloc[i-1] * decay_factor
            combined_signal = combined_signal + decayed_previous
        
        result.iloc[i] = combined_signal
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
