import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate rolling windows for various timeframes
    for i in range(len(data)):
        if i < 20:  # Need at least 20 days of history
            result.iloc[i] = 0
            continue
            
        current = data.iloc[i]
        prev = data.iloc[i-1] if i > 0 else current
        
        # === Regime-Adaptive Reversal Asymmetry ===
        # Volatility Regime Classification
        high_4 = data['high'].iloc[i-4:i+1].max()
        low_4 = data['low'].iloc[i-4:i+1].min()
        sum_range_4 = (data['high'].iloc[i-4:i+1] - data['low'].iloc[i-4:i+1]).sum()
        short_term_vol = (high_4 - low_4) / (sum_range_4 / 5) if sum_range_4 != 0 else 1
        
        if short_term_vol > 1.2:
            regime = "High"
        elif short_term_vol < 0.8:
            regime = "Low"
        else:
            regime = "Normal"
        
        # Asymmetric Reversal Components
        gap_reversal = (current['open'] - prev['close']) / prev['close'] if prev['close'] != 0 else 0
        intraday_recovery = (current['close'] - current['low']) / (current['high'] - current['low']) if (current['high'] - current['low']) != 0 else 0
        recovery_strength = gap_reversal * intraday_recovery
        
        # Regime-Adaptive Signal
        if regime == "High":
            low_5 = data['low'].iloc[i-5:i].min()
            regime_signal_1 = recovery_strength * (current['close'] - low_5) / low_5 if low_5 != 0 else 0
        elif regime == "Low":
            close_10 = data['close'].iloc[i-10]
            regime_signal_1 = recovery_strength * (current['close'] - close_10) / close_10 if close_10 != 0 else 0
        else:
            high_5 = data['high'].iloc[i-5:i].max()
            low_5 = data['low'].iloc[i-5:i].min()
            mid_5 = (high_5 + low_5) / 2
            regime_signal_1 = recovery_strength * (current['close'] - mid_5) / mid_5 if mid_5 != 0 else 0
        
        # === Volume-Price Compression Divergence ===
        # Compression Framework
        volume_compression = current['volume'] / (current['high'] - current['low']) if (current['high'] - current['low']) != 0 else 0
        volatility_compression = (current['high'] - current['low']) / current['close'] if current['close'] != 0 else 0
        compression_ratio = volume_compression * volatility_compression
        
        # Divergence Detection
        volume_momentum = (current['volume'] - prev['volume']) / prev['volume'] if prev['volume'] != 0 else 0
        price_momentum = (current['close'] - prev['close']) / prev['close'] if prev['close'] != 0 else 0
        momentum_divergence = volume_momentum - price_momentum
        
        # Compression-Divergence Signal
        volume_5 = data['volume'].iloc[i-5]
        volume_10 = data['volume'].iloc[i-10]
        vol_accel_1 = (current['volume'] - volume_5) / volume_5 if volume_5 != 0 else 0
        vol_accel_2 = (volume_5 - volume_10) / volume_10 if volume_10 != 0 else 0
        volume_acceleration = vol_accel_1 - vol_accel_2
        
        regime_signal_2 = compression_ratio * momentum_divergence * np.sign(volume_acceleration)
        
        # === Multi-Timeframe Trend Efficiency ===
        # Trend Strength Framework
        high_4_window = data['high'].iloc[i-4:i+1]
        low_4_window = data['low'].iloc[i-4:i+1]
        mid_4 = (high_4_window + low_4_window).mean() / 2
        short_term_trend = (current['close'] - mid_4) / mid_4 if mid_4 != 0 else 0
        
        close_10 = data['close'].iloc[i-10]
        medium_term_trend = (current['close'] - close_10) / close_10 if close_10 != 0 else 0
        
        weighted_trend = 0.7 * short_term_trend + 0.3 * medium_term_trend
        
        # Efficiency Measurement
        daily_efficiency = (current['close'] - current['open']) / (current['high'] - current['low']) if (current['high'] - current['low']) != 0 else 0
        trend_efficiency = weighted_trend * daily_efficiency
        
        # Volume-Confirmed Trend
        volume_trend_concordance = np.sign(weighted_trend) * np.sign(volume_momentum)
        regime_signal_3 = trend_efficiency * volume_trend_concordance
        
        # === Cross-Sectional Volatility Pressure ===
        # Relative Volatility Framework
        high_19 = data['high'].iloc[i-19:i+1].max()
        low_19 = data['low'].iloc[i-19:i+1].min()
        medium_term_range = high_19 - low_19
        volatility_ratio = (high_4 - low_4) / medium_term_range if medium_term_range != 0 else 1
        
        # Price Pressure Components
        high_4_window = data['high'].iloc[i-4:i+1]
        low_4_window = data['low'].iloc[i-4:i+1]
        mid_4_window = (high_4_window + low_4_window).mean() / 2
        relative_strength = current['close'] / mid_4_window if mid_4_window != 0 else 1
        
        pressure_index = (current['high'] - current['close']) / (current['close'] - current['low']) if (current['close'] - current['low']) != 0 else 1
        cross_pressure = relative_strength / pressure_index if pressure_index != 0 else relative_strength
        
        # Volatility-Pressure Signal
        regime_alignment = (volatility_ratio - 1) / 2
        regime_signal_4 = cross_pressure * (1 + abs(regime_alignment))
        
        # === Adaptive Momentum Convergence ===
        # Multi-Regime Momentum
        low_5_window = data['low'].iloc[i-5:i].min()
        high_5_window = data['high'].iloc[i-5:i].max()
        
        high_vol_momentum = (current['close'] - low_5_window) / low_5_window if low_5_window != 0 else 0
        low_vol_momentum = (current['close'] - close_10) / close_10 if close_10 != 0 else 0
        normal_momentum = (current['close'] - (high_5_window + low_5_window)/2) / ((high_5_window + low_5_window)/2) if ((high_5_window + low_5_window)/2) != 0 else 0
        
        # Convergence Framework
        volatility_persistence = (current['high'] - current['low']) / (prev['high'] - prev['low']) if (prev['high'] - prev['low']) != 0 else 1
        
        close_2 = data['close'].iloc[i-2]
        close_4 = data['close'].iloc[i-4]
        acceleration_rate = (current['close'] - close_2) / (close_2 - close_4) if (close_2 - close_4) != 0 else 1
        
        convergence_ratio = volatility_persistence * acceleration_rate
        
        # Adaptive Signal
        volume_confirmation = np.sign(volume_momentum) * np.sign(convergence_ratio)
        regime_signal_5 = convergence_ratio * volume_confirmation * abs(volume_momentum)
        
        # === Combine all regime signals ===
        final_signal = (
            regime_signal_1 + 
            regime_signal_2 + 
            regime_signal_3 + 
            regime_signal_4 + 
            regime_signal_5
        ) / 5
        
        result.iloc[i] = final_signal
    
    return result
