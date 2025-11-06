import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility Components
    # Multi-timeframe Volatility
    data['mtf_vol'] = ((data['close'] - data['close'].shift(5)) / 
                       (data['close'].shift(5) - data['close'].shift(10) + 1e-8) * 
                       (data['high'] - data['low']) / 
                       (data['high'].shift(5) - data['low'].shift(5) + 1e-8))
    
    # Opening Volatility
    data['open_vol'] = ((data['high'] - data['open']) / 
                        (data['open'] - data['low'] + 1e-8) * 
                        (data['close'] - data['close'].shift(5)) / 
                        (data['close'].shift(5) - data['close'].shift(10) + 1e-8))
    
    # Momentum Acceleration for Volatility Expansion
    mom_accel = ((data['close'] - data['close'].shift(3)) / data['close'].shift(3) - 
                 (data['close'].shift(3) - data['close'].shift(6)) / data['close'].shift(6))
    
    # Volatility Expansion
    data['vol_expansion'] = ((data['high'] - data['low']) / 
                            (data['high'].shift(1) - data['low'].shift(1) + 1e-8) * 
                            mom_accel)
    
    # Volume Components
    # Volume-Weighted Momentum
    data['vol_weighted_mom'] = ((data['close'] - data['close'].shift(5)) / 
                               (data['close'].shift(5) - data['close'].shift(10) + 1e-8) * 
                               (data['volume'] / (data['volume'].shift(5) + 1e-8)))
    
    # Trade Size Momentum
    data['trade_size_mom'] = ((data['amount'] / (data['volume'] + 1e-8)) * 
                             (data['close'] - data['close'].shift(5)) / 
                             (data['close'].shift(5) - data['close'].shift(10) + 1e-8))
    
    # Volume-Range Coherence
    data['vol_range_coherence'] = (data['volume'] * (data['high'] - data['low']) / 
                                  (data['volume'].shift(1) * (data['high'].shift(1) - data['low'].shift(1)) + 1e-8))
    
    # Microstructure Components
    # Rejection Signal
    data['upper_shadow'] = data['high'] - np.maximum(data['open'], data['close'])
    data['lower_shadow'] = np.minimum(data['open'], data['close']) - data['low']
    data['rejection_signal'] = ((data['upper_shadow'] - data['lower_shadow']) * 
                               (data['close'] - data['close'].shift(5)) / 
                               (data['close'].shift(5) - data['close'].shift(10) + 1e-8))
    
    # Intraday Efficiency
    data['intraday_efficiency'] = (np.abs(data['close'] - data['open']) / 
                                  (data['high'] - data['low'] + 1e-8) * 
                                  (data['close'] - data['close'].shift(5)) / 
                                  (data['close'].shift(5) - data['close'].shift(10) + 1e-8))
    
    # Order Flow
    data['order_flow'] = (((data['close'] - data['low']) - (data['high'] - data['close'])) / 
                         (data['high'] - data['low'] + 1e-8) * 
                         data['volume'] * 
                         (data['close'] - data['close'].shift(5)) / 
                         (data['close'].shift(5) - data['close'].shift(10) + 1e-8))
    
    # Momentum Acceleration
    # Short-term Acceleration
    data['short_term_accel'] = ((data['close'] - data['close'].shift(3)) / data['close'].shift(3) - 
                               (data['close'].shift(3) - data['close'].shift(6)) / data['close'].shift(6))
    
    # Volume-Confirmed Acceleration
    data['vol_confirmed_accel'] = ((data['close'] - data['close'].shift(1)) / data['close'].shift(1) * 
                                  (data['volume'] / (data['volume'].shift(1) + 1e-8) - 1))
    
    # Persistence Components
    # Momentum Consistency
    momentum_sign = np.sign(data['close'] - data['close'].shift(1))
    target_momentum_sign = np.sign(data['close'] - data['close'].shift(5))
    
    momentum_consistency = pd.Series(index=data.index, dtype=float)
    for i in range(5, len(data)):
        window = momentum_sign.iloc[i-4:i+1]
        consistency_count = (window == target_momentum_sign.iloc[i]).sum()
        momentum_consistency.iloc[i] = consistency_count / 5
    
    # Volume Alignment
    volume_sign = np.sign(data['volume'] - data['volume'].shift(1))
    price_sign = np.sign(data['close'] - data['close'].shift(1))
    
    volume_alignment = pd.Series(index=data.index, dtype=float)
    for i in range(5, len(data)):
        vol_window = volume_sign.iloc[i-4:i+1]
        price_window = price_sign.iloc[i-4:i+1]
        alignment_count = (vol_window == price_window).sum()
        volume_alignment.iloc[i] = alignment_count / 5
    
    # Composite Alpha Construction
    # Core Factor: Volatility Components × Volume Components
    volatility_components = (data['mtf_vol'] + data['open_vol'] + data['vol_expansion']) / 3
    volume_components = (data['vol_weighted_mom'] + data['trade_size_mom'] + data['vol_range_coherence']) / 3
    core_factor = volatility_components * volume_components
    
    # Confirmation Factor: Microstructure Components × Momentum Acceleration
    microstructure_components = (data['rejection_signal'] + data['intraday_efficiency'] + data['order_flow']) / 3
    momentum_acceleration = (data['short_term_accel'] + data['vol_confirmed_accel']) / 2
    confirmation_factor = microstructure_components * momentum_acceleration
    
    # Persistence Components
    persistence_components = (momentum_consistency + volume_alignment) / 2
    
    # Final Alpha
    final_alpha = core_factor * confirmation_factor * persistence_components
    
    # Clean up and return
    final_alpha = final_alpha.replace([np.inf, -np.inf], np.nan)
    return final_alpha
