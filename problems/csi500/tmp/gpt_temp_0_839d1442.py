import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum with Volume-Price Regime Alignment alpha factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic components
    df['intraday_momentum'] = df['close'] - df['open']
    df['daily_range'] = df['high'] - df['low']
    df['range_adjusted_momentum'] = df['intraday_momentum'] / (df['daily_range'] + 1e-8)
    df['volume_weighted_momentum'] = df['intraday_momentum'] * df['volume']
    df['volume_change'] = df['volume'].diff()
    df['volume_direction'] = np.sign(df['volume_change'])
    
    # Multi-timeframe momentum calculations
    for i in range(len(df)):
        if i < 20:  # Need at least 20 days of history
            continue
            
        # Short-term momentum (3-5 days)
        short_term_momentum = df['intraday_momentum'].iloc[i-2:i+1].sum()
        short_term_price_change = df['close'].iloc[i] - df['close'].iloc[i-3]
        short_term_range_avg = df['daily_range'].iloc[i-2:i+1].mean()
        
        # Medium-term momentum (10-20 days)
        medium_term_momentum = df['intraday_momentum'].iloc[i-9:i+1].sum()
        medium_term_price_change = df['close'].iloc[i] - df['close'].iloc[i-10]
        medium_term_range_avg = df['daily_range'].iloc[i-9:i+1].mean()
        
        # Volatility regime
        short_term_vol = df['daily_range'].iloc[i-4:i+1].mean()
        medium_term_vol = df['daily_range'].iloc[i-19:i+1].mean()
        vol_ratio = short_term_vol / (medium_term_vol + 1e-8)
        
        if vol_ratio > 1.15:
            vol_regime = 'high'
        elif vol_ratio >= 0.85:
            vol_regime = 'normal'
        else:
            vol_regime = 'low'
        
        # Volume regime
        short_term_volume = df['volume'].iloc[i-4:i+1].mean()
        medium_term_volume = df['volume'].iloc[i-19:i+1].mean()
        volume_ratio = short_term_volume / (medium_term_volume + 1e-8)
        
        if volume_ratio > 1.1:
            volume_regime = 'high'
        elif volume_ratio >= 0.9:
            volume_regime = 'normal'
        else:
            volume_regime = 'low'
        
        # Momentum regime - direction consistency
        intraday_dir = np.sign(df['intraday_momentum'].iloc[i])
        short_term_dir = np.sign(short_term_momentum)
        medium_term_dir = np.sign(medium_term_momentum)
        
        consistency_score = sum([
            intraday_dir == short_term_dir,
            intraday_dir == medium_term_dir,
            short_term_dir == medium_term_dir
        ])
        
        # Momentum strength
        relative_strength = abs(short_term_momentum) / (short_term_range_avg + 1e-8)
        if relative_strength > 0.03:
            momentum_strength = 'strong'
        elif relative_strength >= 0.01:
            momentum_strength = 'moderate'
        else:
            momentum_strength = 'weak'
        
        # Acceleration
        prev_short_term_momentum = df['intraday_momentum'].iloc[i-4:i-1].sum()
        acceleration = short_term_momentum - prev_short_term_momentum
        
        # Persistence tracking
        # Direction persistence
        current_dir = intraday_dir
        persistence_counter = 1
        for j in range(1, min(10, i+1)):
            if np.sign(df['intraday_momentum'].iloc[i-j]) == current_dir:
                persistence_counter += 1
            else:
                break
        
        persistence_strength = persistence_counter * abs(df['intraday_momentum'].iloc[i])
        
        # Volume-momentum alignment
        alignment_signal = intraday_dir * df['volume_direction'].iloc[i]
        alignment_persistence = 1
        for j in range(1, min(8, i+1)):
            if (np.sign(df['intraday_momentum'].iloc[i-j]) * 
                np.sign(df['volume_change'].iloc[i-j])) > 0:
                alignment_persistence += 1
            else:
                break
        
        alignment_strength = alignment_persistence * abs(df['volume_change'].iloc[i])
        
        # Range persistence
        range_vs_avg = df['daily_range'].iloc[i] / short_term_vol
        range_regime_persistence = 1
        for j in range(1, min(6, i+1)):
            if (df['daily_range'].iloc[i-j] / 
                df['daily_range'].iloc[i-j-4:i-j+1].mean()) > 1.1:
                range_regime_persistence += 1
            else:
                break
        
        range_strength = range_regime_persistence * df['daily_range'].iloc[i]
        
        # Base momentum factor construction
        base_momentum = (
            4 * df['intraday_momentum'].iloc[i] +
            2 * short_term_momentum +
            1 * medium_term_momentum
        )
        
        # Volume integration
        base_factor = base_momentum * df['volume'].iloc[i]
        
        # Persistence enhancement
        base_factor *= (1 + persistence_counter / 8)
        base_factor *= (1 + alignment_persistence / 6)
        base_factor *= (1 + range_regime_persistence / 10)
        
        # Regime-based adjustments
        # Volatility scaling
        if vol_regime == 'high':
            base_factor *= 0.6
        elif vol_regime == 'low':
            base_factor *= 1.4
        else:
            base_factor *= 1.0
        
        # Volume regime scaling
        if volume_regime == 'high':
            base_factor *= 1.3
        elif volume_regime == 'low':
            base_factor *= 0.7
        else:
            base_factor *= 1.0
        
        # Momentum regime scaling
        if consistency_score == 3:
            base_factor *= 1.6
        elif consistency_score == 2:
            base_factor *= 1.0
        else:
            base_factor *= 0.4
        
        # Strength confirmation
        if momentum_strength == 'strong':
            base_factor *= 1.5
        elif momentum_strength == 'weak':
            base_factor *= 0.5
        else:
            base_factor *= 1.0
        
        # Acceleration confirmation
        if acceleration > 0:
            base_factor *= 1.2
        elif acceleration < 0:
            base_factor *= 0.8
        else:
            base_factor *= 1.0
        
        # Range context
        base_factor *= df['daily_range'].iloc[i]
        
        result.iloc[i] = base_factor
    
    # Fill NaN values with 0 for early periods
    result = result.fillna(0)
    
    return result
