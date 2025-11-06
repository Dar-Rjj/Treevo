import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 10:  # Need at least 10 days of data
            result.iloc[i] = 0
            continue
            
        current = df.iloc[i]
        # Get historical data safely
        def safe_get(idx, col):
            if idx < 0:
                return np.nan
            return df.iloc[idx][col]
        
        # Multi-Timeframe Momentum Components
        # Short-Term Microstructure Momentum
        if current['high'] != current['low']:
            short_term_momentum = (current['close'] - current['open']) / (current['high'] - current['low'])
        else:
            short_term_momentum = 0
        
        # Medium-Term Price Momentum
        close_t_minus_5 = safe_get(i-5, 'close')
        close_t_minus_10 = safe_get(i-10, 'close')
        if close_t_minus_5 is not None and close_t_minus_10 is not None and abs(close_t_minus_5 - close_t_minus_10) > 0:
            medium_term_momentum = (current['close'] - close_t_minus_5) / abs(close_t_minus_5 - close_t_minus_10)
        else:
            medium_term_momentum = 0
        
        # Momentum Persistence
        close_t_minus_1 = safe_get(i-1, 'close')
        close_t_minus_3 = safe_get(i-3, 'close')
        if close_t_minus_1 is not None and close_t_minus_3 is not None:
            momentum_persistence = np.sign(current['close'] - close_t_minus_1) * abs(current['close'] - close_t_minus_3)
        else:
            momentum_persistence = 0
        
        # Volatility-Scaled Microstructure
        # Range Volatility Scaling
        range_vol_sum = 0
        range_vol_count = 0
        for j in range(1, 6):  # t-5 to t-1
            hist_high = safe_get(i-j, 'high')
            hist_low = safe_get(i-j, 'low')
            if hist_high is not None and hist_low is not None:
                range_vol_sum += (hist_high - hist_low)
                range_vol_count += 1
        
        if range_vol_count > 0 and range_vol_sum > 0:
            avg_range_vol = range_vol_sum / range_vol_count
            range_vol_scaling = (current['high'] - current['low']) / avg_range_vol
        else:
            range_vol_scaling = 1
        
        # Microstructure Volatility Efficiency
        if (current['high'] - current['low']) > 0 and close_t_minus_1 is not None and abs(current['close'] - close_t_minus_1) > 0:
            microstructure_vol_eff = ((current['close'] - current['open']) ** 2) / ((current['high'] - current['low']) * abs(current['close'] - close_t_minus_1))
        else:
            microstructure_vol_eff = 0
        
        # Volatility-Adjusted Momentum
        if range_vol_scaling > 0:
            volatility_adjusted_momentum = medium_term_momentum / range_vol_scaling
        else:
            volatility_adjusted_momentum = medium_term_momentum
        
        # Volume-Enhanced Microstructure Signals
        # Volume-Momentum Alignment
        volume_t_minus_1 = safe_get(i-1, 'volume')
        volume_t_minus_2 = safe_get(i-2, 'volume')
        
        if volume_t_minus_1 is not None and volume_t_minus_2 is not None:
            avg_volume = (current['volume'] + volume_t_minus_1 + volume_t_minus_2) / 3
            if avg_volume > 0:
                volume_momentum_alignment = (current['volume'] / avg_volume) * np.sign(short_term_momentum)
            else:
                volume_momentum_alignment = 0
        else:
            volume_momentum_alignment = 0
        
        # Microstructure Pressure Indicator
        if current['amount'] > 0:
            microstructure_pressure = (current['high'] - current['low']) * current['volume'] / current['amount']
        else:
            microstructure_pressure = 0
        
        # Opening Efficiency
        if current['amount'] > 0 and close_t_minus_1 is not None:
            opening_efficiency = abs(current['open'] - close_t_minus_1) * current['volume'] / current['amount']
        else:
            opening_efficiency = 0
        
        # Price Range Dynamics
        # Intraday Range Quality
        if current['high'] != current['low']:
            intraday_range_quality = (current['close'] - current['open']) / (current['high'] - current['low'])
        else:
            intraday_range_quality = 0
        
        # Range Persistence Ratio
        high_t_minus_1 = safe_get(i-1, 'high')
        low_t_minus_1 = safe_get(i-1, 'low')
        if high_t_minus_1 is not None and low_t_minus_1 is not None and (high_t_minus_1 - low_t_minus_1) > 0:
            range_persistence_ratio = (current['high'] - current['low']) / (high_t_minus_1 - low_t_minus_1)
        else:
            range_persistence_ratio = 1
        
        # Asymmetric Range Momentum
        if (current['close'] - current['low']) > 0:
            asymmetric_range_momentum = ((current['high'] - current['close']) / (current['close'] - current['low'])) * np.sign(medium_term_momentum)
        else:
            asymmetric_range_momentum = 0
        
        # Cross-Dimension Integration
        # Volatility-Volume Convergence
        volatility_volume_convergence = range_vol_scaling * volume_momentum_alignment
        
        # Microstructure-Momentum Quality
        microstructure_momentum_quality = microstructure_vol_eff * momentum_persistence
        
        # Range-Momentum Efficiency
        range_momentum_efficiency = intraday_range_quality * volatility_adjusted_momentum
        
        # Composite Factor Construction
        # Core Momentum Signal (weighted average)
        core_momentum_signal = (
            0.4 * volatility_adjusted_momentum +
            0.3 * short_term_momentum +
            0.3 * momentum_persistence
        )
        
        # Microstructure Quality Filter (multiplicative combination)
        microstructure_quality_filter = (
            microstructure_vol_eff *
            intraday_range_quality *
            range_persistence_ratio
        )
        
        # Volume confidence adjustment
        volume_confidence = np.tanh(current['volume'] / max(1, avg_volume if 'avg_volume' in locals() else current['volume']))
        
        # Final Dynamic Factor
        final_factor = core_momentum_signal * microstructure_quality_filter * volume_confidence
        
        result.iloc[i] = final_factor
    
    # Handle any remaining NaN values
    result = result.fillna(0)
    
    return result
