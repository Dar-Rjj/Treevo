import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Quantum Price Dynamics
    data['quantum_price_echo'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['price_wave_interference'] = ((data['high'] - data['close']) * (data['close'] - data['low'])) / ((data['high'] - data['low']) ** 2)
    
    # Quantum Entanglement with safe division
    close_diff = data['close'] - data['close'].shift(1)
    volume_diff = data['volume'] - data['volume'].shift(1)
    high_low_range = data['high'] - data['low']
    data['quantum_entanglement'] = (close_diff * volume_diff) / high_low_range.replace(0, np.nan)
    
    # Cross-Asset Quantum States (simplified - using rolling correlation with close prices)
    # For demonstration, we'll use a rolling correlation of stock returns with itself (sector proxy)
    data['stock_3d_return'] = data['close'].pct_change(3)
    # Using rolling correlation with lagged returns as sector proxy
    data['sector_quantum_coherence'] = data['stock_3d_return'].rolling(window=15).corr(data['stock_3d_return'].shift(3))
    
    # Asset Superposition (simplified - using sign of returns)
    data['asset_superposition'] = (np.sign(data['stock_3d_return']) * 
                                  np.sign(data['stock_3d_return'].shift(3)) * 
                                  np.sign(data['stock_3d_return'].shift(6)))
    
    # Microstructure Quantum Patterns
    data['quantum_tunneling'] = np.abs(data['close'] - (data['high'] + data['low'])/2) / (data['high'] - data['low'])
    
    # Momentum Quantum Leap with safe division
    mom_2d = data['close'] / data['close'].shift(2) - 1
    mom_4d = data['close'] / data['close'].shift(4) - 1
    data['momentum_quantum_leap'] = mom_2d / mom_4d.replace(0, np.nan)
    
    # Quantum Liquidity Flow
    # Volume Quantum Persistence
    vol_ma_4 = data['volume'].rolling(window=4).mean().shift(1)
    data['volume_quantum_persistence'] = 0
    for i in range(1, len(data)):
        if data['volume'].iloc[i] < vol_ma_4.iloc[i]:
            data.loc[data.index[i], 'volume_quantum_persistence'] = data['volume_quantum_persistence'].iloc[i-1] + 1
    
    # Microstructure Quantum Field
    data['microstructure_quantum_field'] = ((data['close'] - data['open']) / (data['high'] - data['low'])) * (data['volume'] / data['volume'].shift(1))
    
    # Quantum Alpha Synthesis
    # Quantum Base Signal
    data['quantum_base_signal'] = (data['quantum_price_echo'] * 
                                  data['microstructure_quantum_field'] * 
                                  data['sector_quantum_coherence'])
    
    # Quantum Enhancement
    data['quantum_enhancement'] = (data['quantum_base_signal'] * 
                                  (1 + np.abs(data['price_wave_interference']) * np.abs(data['quantum_entanglement'])))
    
    # Final Quantum Alpha
    data['quantum_alpha'] = (data['quantum_enhancement'] * 
                            data['momentum_quantum_leap'] * 
                            data['volume_quantum_persistence'])
    
    # Return the final alpha factor
    return data['quantum_alpha']
