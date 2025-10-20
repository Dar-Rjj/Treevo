import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Asymmetric Volume-Price Efficiency Factor
    Combines directional price efficiency with volume asymmetry to detect regime shifts
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Directional Price Efficiency
    data['price_range'] = data['high'] - data['low']
    data['price_range'] = np.where(data['price_range'] == 0, 1e-6, data['price_range'])  # Avoid division by zero
    
    # Up-day efficiency
    up_mask = data['close'] > data['open']
    data['up_efficiency'] = np.where(up_mask, 
                                   (data['close'] - data['open']) / data['price_range'], 
                                   0)
    
    # Down-day efficiency  
    down_mask = data['close'] < data['open']
    data['down_efficiency'] = np.where(down_mask, 
                                     (data['open'] - data['close']) / data['price_range'], 
                                     0)
    
    # Measure Volume Asymmetry
    # 10-day rolling average volume
    data['avg_volume_10d'] = data['volume'].rolling(window=10, min_periods=5).mean()
    
    # Up-volume intensity
    data['up_volume_intensity'] = np.where(up_mask, 
                                         data['volume'] / data['avg_volume_10d'], 
                                         0)
    
    # Down-volume intensity
    data['down_volume_intensity'] = np.where(down_mask, 
                                           data['volume'] / data['avg_volume_10d'], 
                                           0)
    
    # Compute Efficiency-Volume Divergence
    # Rolling averages for efficiency and volume intensity
    data['up_eff_5d'] = data['up_efficiency'].rolling(window=5, min_periods=3).mean()
    data['up_vol_int_5d'] = data['up_volume_intensity'].rolling(window=5, min_periods=3).mean()
    data['down_eff_5d'] = data['down_efficiency'].rolling(window=5, min_periods=3).mean()
    data['down_vol_int_5d'] = data['down_volume_intensity'].rolling(window=5, min_periods=3).mean()
    
    # Divergence measures
    data['up_divergence'] = data['up_eff_5d'] - data['up_vol_int_5d']
    data['down_divergence'] = data['down_eff_5d'] - data['down_vol_int_5d']
    
    # Detect Market Regime Shifts using rolling correlation
    data['eff_vol_corr_20d'] = data['up_efficiency'].rolling(window=20, min_periods=10).corr(data['up_volume_intensity'])
    
    # Generate Alpha Signal
    # High up-efficiency + low up-volume → potential continuation
    # Low down-efficiency + high down-volume → potential reversal
    
    # Signal components
    up_continuation = np.where((data['up_eff_5d'] > data['up_eff_5d'].rolling(window=20).mean()) & 
                              (data['up_vol_int_5d'] < data['up_vol_int_5d'].rolling(window=20).mean()), 1, 0)
    
    down_reversal = np.where((data['down_eff_5d'] < data['down_eff_5d'].rolling(window=20).mean()) & 
                            (data['down_vol_int_5d'] > data['down_vol_int_5d'].rolling(window=20).mean()), -1, 0)
    
    # Regime shift detection
    regime_shift = np.where(data['eff_vol_corr_20d'] < data['eff_vol_corr_20d'].rolling(window=40).mean() - 
                           data['eff_vol_corr_20d'].rolling(window=40).std(), 1, 0)
    
    # Combined alpha signal
    alpha_signal = (up_continuation + down_reversal) * (1 + 0.5 * regime_shift)
    
    # Normalize the signal
    alpha_signal_normalized = (alpha_signal - alpha_signal.rolling(window=60).mean()) / alpha_signal.rolling(window=60).std()
    
    return alpha_signal_normalized
