import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Ensure proper indexing and calculate necessary intermediate values
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    data['prev_amount'] = data['amount'].shift(1)
    
    # Handle NaN values from shifting
    data = data.fillna(method='bfill')
    
    # Intraday Volatility Dynamics
    # Opening Volatility Fractures
    gap_volatility_fracture = ((data['open'] - data['prev_close']) / 
                              (data['prev_high'] - data['prev_low'])) * (data['volume'] / data['prev_volume'])
    
    opening_range_efficiency = ((data['high'] - data['low']) / 
                               np.abs(data['open'] - data['prev_close'])) * (data['amount'] / data['prev_amount'])
    
    morning_momentum_fracture = ((data['close'] - data['open']) / 
                                (data['high'] - data['low'])) * np.sign(data['volume'] - data['prev_volume'])
    
    # Session Volatility Structure
    volatility_compression = ((data['high'] - data['low']) / 
                             (data['prev_high'] - data['prev_low'])) * (data['volume'] / data['prev_volume'])
    
    # Range persistence (5-day lookback)
    range_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(4, len(data)):
        window = data.iloc[i-4:i+1]
        count = sum((window['high'] - window['low']) < (window['high'].shift(1) - window['low'].shift(1)))
        range_persistence.iloc[i] = count
    
    volatility_volume_alignment = ((data['high'] - data['low']) * data['volume'] / 
                                  ((data['prev_high'] - data['prev_low']) * data['prev_volume']))
    
    # Closing Volatility Patterns
    afternoon_momentum_shift = (((data['close'] - (data['high'] + data['low'])/2) / 
                               (data['high'] - data['low'])) * (data['amount'] / data['prev_amount'] - 1))
    
    end_of_day_volatility = (np.abs(data['close'] - data['open']) / 
                            (data['high'] - data['low'])) * (data['volume'] / data['prev_volume'])
    
    closing_efficiency_fracture = ((data['close'] - data['prev_close']) / 
                                  (data['high'] - data['low'])) * np.sign(data['amount'] - data['prev_amount'])
    
    # Regime Transition Detection
    # Volatility Breakout Signals
    high_vol_breakout = np.where((data['high'] - data['low']) > 1.5 * (data['prev_high'] - data['prev_low']),
                                data['volume'] / data['prev_volume'], 0)
    
    low_vol_compression = np.where((data['high'] - data['low']) < 0.7 * (data['prev_high'] - data['prev_low']),
                                  data['amount'] / data['prev_amount'], 0)
    
    breakout_confirmation = high_vol_breakout * low_vol_compression
    
    # Volume-Volatility Divergence
    high_volume_low_vol = (data['volume'] / data['prev_volume']) / ((data['high'] - data['low']) / (data['prev_high'] - data['prev_low']))
    low_volume_high_vol = ((data['high'] - data['low']) / (data['prev_high'] - data['prev_low'])) / (data['volume'] / data['prev_volume'])
    
    # Divergence persistence
    divergence_persistence = pd.Series(index=data.index, dtype=float)
    consecutive_count = 0
    for i in range(len(data)):
        if high_volume_low_vol[i] > 1.2:
            consecutive_count += 1
        else:
            consecutive_count = 0
        divergence_persistence.iloc[i] = consecutive_count
    
    # Price Efficiency in Volatility
    volatility_adjusted_momentum = ((data['close'] - data['prev_close']) / 
                                   (data['high'] - data['low'])) * (data['volume'] / data['prev_volume'])
    
    range_utilization_efficiency = (np.abs(data['close'] - data['open']) / 
                                  (data['high'] - data['low'])) * (data['amount'] / data['prev_amount'])
    
    efficiency_vol_alignment = volatility_adjusted_momentum * range_utilization_efficiency
    
    # Multi-Timeframe Volatility Structure
    # Short-term patterns
    intraday_vol_momentum = ((data['high'] - data['low']) / (data['prev_high'] - data['prev_low']) - 
                            (data['high'].shift(2) - data['low'].shift(2)) / (data['high'].shift(3) - data['low'].shift(3)))
    
    volume_vol_correlation = (np.sign(data['volume'] - data['prev_volume']) * 
                            np.sign((data['high'] - data['low']) - (data['prev_high'] - data['prev_low'])))
    
    short_term_efficiency = (np.abs(data['close'] - data['prev_close']) / 
                           (data['high'] - data['low'])) * (data['volume'] / data['prev_volume'])
    
    # Medium-term patterns (5-day)
    vol_trend_5d = (data['high'] - data['low']) / data['high'].rolling(window=5).mean()
    
    volume_regime_alignment = ((data['volume'] / data['volume'].rolling(window=5).mean()) * 
                              (data['high'] - data['low']) / (data['high'].rolling(window=5).mean() - data['low'].rolling(window=5).mean()))
    
    medium_term_efficiency = (np.abs(data['close'] - data['close'].shift(4)) / 
                            (data['high'] - data['low'])) * (data['amount'] / data['amount'].rolling(window=5).mean())
    
    # Volatility Structure Integration
    multi_timeframe_alignment = short_term_efficiency * medium_term_efficiency
    vol_regime_consistency = volume_vol_correlation * volume_regime_alignment
    structural_efficiency = multi_timeframe_alignment * vol_regime_consistency
    
    # Fracture Point Identification
    # Critical Volatility Levels
    vol_ratio = (data['high'] - data['low']) / (data['prev_high'] - data['prev_low'])
    high_vol_fracture = np.where(vol_ratio > 1.8, data['volume'] / data['prev_volume'], 1.0)
    low_vol_fracture = np.where(vol_ratio < 0.5, data['amount'] / data['prev_amount'], 1.0)
    vol_regime_multiplier = np.where(vol_ratio > 1.8, 1.4, np.where(vol_ratio < 0.5, 0.6, 1.0))
    
    # Volume Break Points
    vol_ratio_vol = data['volume'] / data['prev_volume']
    volume_surge_fracture = np.where(vol_ratio_vol > 2.0, vol_ratio, 1.0)
    volume_collapse_fracture = np.where(vol_ratio_vol < 0.5, (data['close'] - data['prev_close']) / (data['high'] - data['low']), 1.0)
    volume_regime_multiplier = np.where(vol_ratio_vol > 2.0, 1.2, np.where(vol_ratio_vol < 0.5, 0.8, 1.0))
    
    # Efficiency Thresholds
    efficiency_regime_multiplier = np.where(range_utilization_efficiency > 0.7, 1.3, 
                                          np.where(range_utilization_efficiency < 0.3, 0.7, 1.0))
    
    # Composite Volatility Alpha
    # Core Volatility Signal
    base_vol_momentum = volatility_adjusted_momentum * volume_vol_correlation
    range_efficiency_enhancement = base_vol_momentum * range_utilization_efficiency
    multi_timeframe_refinement = range_efficiency_enhancement * multi_timeframe_alignment
    
    # Fracture Dynamics Integration
    breakout_amplification = np.where(high_vol_breakout > 0, 1.25, 1.0)
    compression_enhancement = np.where(low_vol_compression > 0, 1.15, 1.0)
    fracture_dynamics = breakout_amplification * compression_enhancement
    
    # Final Alpha Construction
    regime_weighted_base = multi_timeframe_refinement * ((vol_regime_multiplier + volume_regime_multiplier + efficiency_regime_multiplier) / 3)
    fracture_adjusted_signal = regime_weighted_base * fracture_dynamics
    final_vol_regime_alpha = fracture_adjusted_signal * structural_efficiency
    
    # Normalize and return
    alpha_series = pd.Series(final_vol_regime_alpha, index=data.index)
    alpha_series = alpha_series.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    
    return alpha_series
