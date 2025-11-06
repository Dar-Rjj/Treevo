import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a composite alpha factor using asymmetric price-volume dynamics,
    multi-timeframe volatility regimes, and pattern-based asymmetric signals.
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate rolling windows and required intermediate calculations
    for i in range(len(df)):
        if i < 20:  # Need at least 20 days for some calculations
            alpha.iloc[i] = 0
            continue
            
        current = df.iloc[i]
        
        # Get historical data
        close_t = current['close']
        open_t = current['open']
        high_t = current['high']
        low_t = current['low']
        volume_t = current['volume']
        amount_t = current['amount']
        
        # Previous day data
        prev_day = df.iloc[i-1]
        close_t1 = prev_day['close']
        high_t1 = prev_day['high']
        low_t1 = prev_day['low']
        volume_t1 = prev_day['volume']
        
        # 5-day lookback data
        if i >= 5:
            close_t5 = df.iloc[i-5]['close']
            high_t5 = df.iloc[i-5]['high']
            low_t5 = df.iloc[i-5]['low']
            volume_t5 = df.iloc[i-5]['volume']
        else:
            close_t5 = close_t
            high_t5 = high_t
            low_t5 = low_t
            volume_t5 = volume_t
        
        # 20-day lookback data
        close_t20 = df.iloc[i-20]['close']
        
        # Rolling windows for calculations
        close_3d = df.iloc[max(0, i-3):i+1]['close']
        close_10d = df.iloc[max(0, i-10):i+1]['close']
        close_5d_window = df.iloc[max(0, i-5):i+1]['close']
        volume_5d_window = df.iloc[max(0, i-5):i+1]['volume']
        
        # Core Volatility-Regime Factor components
        # Regime directional ratio
        if (high_t1 - low_t1) > 0 and volume_t5 > 0:
            regime_directional = (close_t - open_t) * (high_t - low_t) / (high_t1 - low_t1) * volume_t / volume_t5
        else:
            regime_directional = 0
        
        # Multi-timeframe convergence components
        # Short-term position asymmetry
        if len(close_3d) >= 2:
            min_close_3d = close_3d.min()
            max_close_3d = close_3d.max()
            if (max_close_3d - min_close_3d) > 0:
                short_term_asymmetry = (close_t - min_close_3d) / (max_close_3d - close_t) * (high_t - low_t) / close_t
            else:
                short_term_asymmetry = 0
        else:
            short_term_asymmetry = 0
        
        # Medium-term range efficiency
        if len(close_10d) >= 2 and i >= 5:
            min_close_10d = close_10d.min()
            max_close_10d = close_10d.max()
            
            # Calculate 5-day volatility
            if len(close_5d_window) >= 2:
                vol_5d = sum(abs(close_5d_window.iloc[j] - close_5d_window.iloc[j-1]) 
                           for j in range(1, len(close_5d_window)))
            else:
                vol_5d = abs(close_t - close_t5)
            
            if (max_close_10d - min_close_10d) > 0 and vol_5d > 0:
                medium_term_efficiency = (close_t - min_close_10d) / (max_close_10d - close_t) * (close_t - close_t5) / vol_5d
            else:
                medium_term_efficiency = 0
        else:
            medium_term_efficiency = 0
        
        # Volume momentum for convergence
        if len(volume_5d_window) >= 2 and volume_t5 > 0:
            volume_momentum = (volume_t / volume_t5 - 1)
        else:
            volume_momentum = 0
        
        multi_timeframe_convergence = short_term_asymmetry * medium_term_efficiency * volume_momentum
        
        # Core Volatility-Regime Factor
        core_volatility_factor = regime_directional * multi_timeframe_convergence
        
        # Price-Volume Microstructure components
        # Signed volume acceleration
        if volume_t5 > 0:
            volume_acceleration = (volume_t / volume_t5 - 1)
            if abs(close_t - open_t) > 0:
                volume_acceleration *= (close_t - open_t) / abs(close_t - open_t)
            else:
                volume_acceleration *= 0
        else:
            volume_acceleration = 0
        
        # Trade size momentum
        if volume_t > 0 and (high_t - low_t) > 0:
            avg_trade_size = amount_t / volume_t
            trade_size_momentum = avg_trade_size * (close_t - close_t1) / (high_t - low_t)
        else:
            trade_size_momentum = 0
        
        price_volume_microstructure = volume_acceleration * trade_size_momentum
        
        # Pattern Efficiency Signal components
        # Directional fractal efficiency
        if len(close_5d_window) >= 2:
            vol_5d = sum(abs(close_5d_window.iloc[j] - close_5d_window.iloc[j-1]) 
                       for j in range(1, len(close_5d_window)))
            if vol_5d > 0:
                directional_efficiency = (close_t - close_t5) / vol_5d
                if abs(close_t - open_t) > 0:
                    directional_efficiency *= (close_t - open_t) / abs(close_t - open_t)
                else:
                    directional_efficiency *= 0
            else:
                directional_efficiency = 0
        else:
            directional_efficiency = 0
        
        # Volatility-adjusted trend
        if len(close_5d_window) >= 2:
            trend_strength = sum(np.sign(close_5d_window.iloc[j] - close_5d_window.iloc[j-1]) 
                               for j in range(1, len(close_5d_window))) / (len(close_5d_window) - 1)
            
            if (high_t5 - low_t5) > 0:
                volatility_adjusted_trend = trend_strength * (high_t - low_t) / (high_t5 - low_t5)
            else:
                volatility_adjusted_trend = 0
        else:
            volatility_adjusted_trend = 0
        
        pattern_efficiency_signal = directional_efficiency * volatility_adjusted_trend
        
        # Final Alpha composition
        final_alpha = core_volatility_factor * price_volume_microstructure * pattern_efficiency_signal
        
        # Handle edge cases and ensure finite values
        if np.isfinite(final_alpha):
            alpha.iloc[i] = final_alpha
        else:
            alpha.iloc[i] = 0
    
    return alpha
