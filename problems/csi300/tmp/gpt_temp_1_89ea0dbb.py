import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Fractal Divergence Factor
    Combines price fractal patterns, volume fractal dynamics, and multi-timeframe divergence
    to create a novel alpha factor.
    """
    df = df.copy()
    
    # Helper function to identify price fractals
    def identify_price_fractals(high_series, low_series, window=2):
        """Identify high and low fractals using local maxima/minima"""
        high_fractals = pd.Series(False, index=high_series.index)
        low_fractals = pd.Series(False, index=low_series.index)
        
        for i in range(window, len(high_series) - window):
            # High fractal: current high is higher than window highs on both sides
            if all(high_series.iloc[i] > high_series.iloc[i-j] for j in range(1, window+1)) and \
               all(high_series.iloc[i] > high_series.iloc[i+j] for j in range(1, window+1)):
                high_fractals.iloc[i] = True
            
            # Low fractal: current low is lower than window lows on both sides
            if all(low_series.iloc[i] < low_series.iloc[i-j] for j in range(1, window+1)) and \
               all(low_series.iloc[i] < low_series.iloc[i+j] for j in range(1, window+1)):
                low_fractals.iloc[i] = True
                
        return high_fractals, low_fractals
    
    # Helper function to identify volume fractals
    def identify_volume_fractals(volume_series, window=2):
        """Identify volume peak and valley fractals"""
        volume_peaks = pd.Series(False, index=volume_series.index)
        volume_valleys = pd.Series(False, index=volume_series.index)
        
        for i in range(window, len(volume_series) - window):
            # Volume peak: current volume is higher than window volumes on both sides
            if all(volume_series.iloc[i] > volume_series.iloc[i-j] for j in range(1, window+1)) and \
               all(volume_series.iloc[i] > volume_series.iloc[i+j] for j in range(1, window+1)):
                volume_peaks.iloc[i] = True
            
            # Volume valley: current volume is lower than window volumes on both sides
            if all(volume_series.iloc[i] < volume_series.iloc[i-j] for j in range(1, window+1)) and \
               all(volume_series.iloc[i] < volume_series.iloc[i+j] for j in range(1, window+1)):
                volume_valleys.iloc[i] = True
                
        return volume_peaks, volume_valleys
    
    # Calculate price fractals
    high_fractals, low_fractals = identify_price_fractals(df['high'], df['low'])
    
    # Calculate volume fractals
    volume_peaks, volume_valleys = identify_volume_fractals(df['volume'])
    
    # Price Fractal Patterns
    # High Fractal Ratio
    high_fractal_count = high_fractals.rolling(window=10, min_periods=1).sum()
    low_fractal_count = low_fractals.rolling(window=10, min_periods=1).sum()
    high_fractal_ratio = high_fractal_count / (low_fractal_count + 1e-8)
    
    # Fractal Momentum
    def get_recent_fractal_levels(close_series, high_fractals, low_fractals):
        """Get recent fractal high and low levels"""
        fractal_highs = []
        fractal_lows = []
        
        for i in range(len(close_series)):
            # Look back for recent fractals (max 20 days)
            recent_highs = []
            recent_lows = []
            
            for j in range(max(0, i-20), i):
                if high_fractals.iloc[j]:
                    recent_highs.append(df['high'].iloc[j])
                if low_fractals.iloc[j]:
                    recent_lows.append(df['low'].iloc[j])
            
            # Use most recent values or current price if none found
            if recent_highs:
                fractal_highs.append(recent_highs[-1])
            else:
                fractal_highs.append(close_series.iloc[i])
                
            if recent_lows:
                fractal_lows.append(recent_lows[-1])
            else:
                fractal_lows.append(close_series.iloc[i])
                
        return pd.Series(fractal_highs, index=close_series.index), pd.Series(fractal_lows, index=close_series.index)
    
    recent_fractal_highs, recent_fractal_lows = get_recent_fractal_levels(df['close'], high_fractals, low_fractals)
    fractal_momentum = (df['close'] - recent_fractal_lows) / (recent_fractal_highs - recent_fractal_lows + 1e-8)
    
    # Fractal Breakout Signal
    current_fractal_count = (high_fractals | low_fractals).rolling(window=5).sum()
    avg_fractal_count = (high_fractals | low_fractals).rolling(window=10).mean()
    fractal_breakout_signal = current_fractal_count / (avg_fractal_count + 1e-8)
    
    # Volume Fractal Dynamics
    # Volume Fractal Asymmetry
    volume_peak_count = volume_peaks.rolling(window=10, min_periods=1).sum()
    volume_valley_count = volume_valleys.rolling(window=10, min_periods=1).sum()
    volume_fractal_asymmetry = volume_peak_count / (volume_valley_count + 1e-8)
    
    # Fractal Co-occurrence
    price_volume_cooccurrence = ((high_fractals | low_fractals) & (volume_peaks | volume_valleys)).rolling(window=10).mean()
    
    # Multi-timeframe Divergence
    # Short-term vs Medium-term fractal count difference
    short_term_fractals = (high_fractals | low_fractals).rolling(window=5).sum()
    medium_term_fractals = (high_fractals | low_fractals).rolling(window=15).sum()
    timeframe_divergence = short_term_fractals - medium_term_fractals
    
    # Fractal pattern persistence
    fractal_persistence = (high_fractals | low_fractals).rolling(window=10).sum() / 10.0
    
    # Composite Alpha
    # Core Divergence
    core_divergence = (fractal_momentum * volume_fractal_asymmetry) / (price_volume_cooccurrence + 1e-8)
    
    # Multi-timeframe Divergence component
    multi_timeframe_component = timeframe_divergence * fractal_persistence
    
    # Final Factor
    final_factor = core_divergence * multi_timeframe_component * fractal_breakout_signal
    
    # Normalize and handle edge cases
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan)
    final_factor = final_factor.fillna(method='ffill').fillna(0)
    
    # Remove any remaining extreme values
    final_factor = np.clip(final_factor, -10, 10)
    
    return final_factor
