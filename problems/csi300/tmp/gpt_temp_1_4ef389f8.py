import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Amplitude-Momentum Acceleration Divergence with Regime-Adaptive Confirmation
    """
    data = df.copy()
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    data['prev_close_3'] = data['close'].shift(3)
    
    # 1. Compute Amplitude-Based Momentum Acceleration
    # Bidirectional Amplitude Momentum
    data['amplitude_up_momentum'] = ((data['high'] - data['close']) * data['volume'] * 
                                   (data['close'] - data['prev_close_3']))
    data['amplitude_down_momentum'] = ((data['close'] - data['low']) * data['volume'] * 
                                     (data['prev_close_3'] - data['close']))
    
    # Net amplitude momentum
    data['net_amplitude_momentum'] = data['amplitude_up_momentum'] - data['amplitude_down_momentum']
    
    # Amplitude acceleration divergence
    data['daily_range'] = data['high'] - data['low']
    data['amplitude_3day'] = data['daily_range'].rolling(window=3, min_periods=3).mean()
    data['amplitude_6day'] = data['daily_range'].rolling(window=6, min_periods=6).mean()
    data['amplitude_acceleration_gap'] = data['amplitude_3day'] - data['amplitude_6day']
    
    # 2. Analyze Volatility-Regime Adjusted Signals
    # True Range calculation
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['atr_10'] = data['true_range'].rolling(window=10, min_periods=10).mean()
    
    # Amplitude regime ratio
    data['amplitude_5day'] = data['daily_range'].rolling(window=5, min_periods=5).mean()
    data['amplitude_20day'] = data['daily_range'].rolling(window=20, min_periods=20).mean()
    data['amplitude_regime_ratio'] = data['amplitude_5day'] / data['amplitude_20day']
    
    # Volume-price confirmation
    data['reversal_magnitude'] = abs(data['close'] - data['prev_close']) / (data['high'] - data['low'] + 1e-8)
    data['reversal_persistence'] = data['reversal_magnitude'].rolling(window=3, min_periods=3).mean()
    
    data['volume_intensity'] = data['volume'] / (data['high'] - data['low'] + 1e-8)
    data['volume_intensity_median_10'] = data['volume_intensity'].rolling(window=10, min_periods=10).median()
    data['volume_intensity_ratio'] = data['volume_intensity'] / (data['volume_intensity_median_10'] + 1e-8)
    
    # 3. Combine with Dynamic Regime Weighting
    # Volatility scaling
    data['volatility_scaled_momentum'] = data['net_amplitude_momentum'] * data['atr_10']
    
    # Amplitude regime multiplier
    data['amplitude_regime_multiplier'] = data['amplitude_regime_ratio']
    
    # Volume-price confirmation weighting
    data['volume_weighted_reversal'] = data['reversal_persistence'] * data['volume_intensity_ratio']
    
    # 4. Regime-Adaptive Signal Integration
    # Volatility regime classification
    data['volatility_regime'] = np.where(data['atr_10'] > data['atr_10'].rolling(window=20, min_periods=20).median(), 
                                       'high', 'low')
    
    # Regime-specific weighting
    data['regime_weight'] = np.where(data['volatility_regime'] == 'high', 
                                   data['volatility_scaled_momentum'] * 1.2,
                                   data['volume_weighted_reversal'] * 0.8)
    
    # Combine amplitude and acceleration components
    data['amplitude_acceleration_interaction'] = (data['net_amplitude_momentum'] * 
                                                data['amplitude_acceleration_gap'] * 
                                                data['amplitude_regime_multiplier'])
    
    # Volume intensity final adjustment
    data['volume_trend'] = data['volume'].rolling(window=3, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 3 else np.nan
    )
    data['volume_confirmation'] = np.where(data['volume_trend'] > 0, 1.1, 0.9)
    
    # 5. Generate Final Adaptive Alpha Factor
    # Composite signal calculation
    data['composite_signal'] = (data['regime_weight'] * 
                              data['amplitude_acceleration_interaction'] * 
                              data['volume_confirmation'])
    
    # Apply cubic root transformation
    data['final_alpha'] = np.sign(data['composite_signal']) * np.cbrt(np.abs(data['composite_signal']))
    
    return data['final_alpha']
