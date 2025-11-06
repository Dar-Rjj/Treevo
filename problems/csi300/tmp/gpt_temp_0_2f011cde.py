import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Add small epsilon to avoid division by zero
    eps = 0.001
    
    # Fractal Price Efficiency components
    # Fractal Momentum Efficiency
    fractal_momentum = ((data['close'] - data['close'].shift(1)) / 
                       (data['close'].shift(1) - data['close'].shift(2) + eps) * 
                       (data['close'] - data['close'].shift(3)) / 
                       (np.abs(data['close'] - data['close'].shift(3)) + eps) * 
                       (data['close'] - data['open']) / 
                       (data['high'] - data['low'] + eps))
    
    # Range Efficiency Momentum
    range_efficiency = ((data['close'] - data['close'].shift(3)) / 
                       (data['high'] - data['low'] + eps) * 
                       (data['close'] - data['open']) / 
                       (np.abs(data['open'] - data['close'].shift(1)) + eps))
    
    # Microstructure Liquidity Integration components
    # Volatility-Adjusted Volume Flow
    vol_adjusted_flow = (((data['close'] - data['low']) / (data['high'] - data['low'] + eps)) * 
                        ((data['volume'] - data['volume'].shift(1)) / 
                         (data['volume'] + data['volume'].shift(1) + eps)) * 
                        np.sign(data['close'] - data['close'].shift(1)) * 
                        np.sign(data['volume'] - data['volume'].shift(1)))
    
    # Volume Distribution Efficiency
    volume_distribution = ((data['volume'] * (data['open'] - data['low']) / (data['high'] - data['low'] + eps) - 
                          data['volume'] * (data['high'] - data['close']) / (data['high'] - data['low'] + eps)) * 
                          np.sign(data['close'] - data['close'].shift(1)))
    
    # Regime-Adaptive Divergence components
    # Price-Efficiency Divergence
    price_efficiency_div = ((data['close'] - data['close'].shift(2)) / (data['close'].shift(2) + eps) - 
                           (data['close'] - data['open']) / (data['high'] - data['low'] + eps))
    
    # Volume-Flow Divergence
    volume_flow_div = (np.sign(data['volume'] - data['volume'].shift(1)) * 
                      np.sign(data['close'] - data['close'].shift(1)) - 
                      np.sign(data['volume'].shift(2) - data['volume'].shift(3)) * 
                      np.sign(data['close'].shift(2) - data['close'].shift(3)))
    
    # Multi-Timeframe Efficiency components
    # Intraday Efficiency Change
    intraday_efficiency_change = ((data['close'] - data['open']) / (data['high'] - data['low'] + eps) - 
                                 (data['close'].shift(1) - data['open'].shift(1)) / 
                                 (data['high'].shift(1) - data['low'].shift(1) + eps))
    
    # Volume Flow Consistency (count same sign over t-3 to t)
    def count_same_sign(window):
        if len(window) < 4:
            return np.nan
        signs = []
        for i in range(len(window)-1):
            price_sign = np.sign(window.iloc[i+1]['close'] - window.iloc[i]['close'])
            volume_sign = np.sign(window.iloc[i+1]['volume'] - window.iloc[i]['volume'])
            signs.append(price_sign * volume_sign)
        # Count how many consecutive periods have the same sign product
        if len(signs) == 0:
            return 0
        current_sign = signs[0]
        count = 1
        for i in range(1, len(signs)):
            if signs[i] == current_sign:
                count += 1
            else:
                break
        return count
    
    # Calculate rolling volume flow consistency
    volume_flow_consistency = pd.Series(index=data.index, dtype=float)
    for i in range(3, len(data)):
        window = data.iloc[i-3:i+1][['close', 'volume']]
        volume_flow_consistency.iloc[i] = count_same_sign(window)
    
    # Composite Alpha Construction
    # Core Efficiency Factor
    core_efficiency = fractal_momentum * range_efficiency
    
    # Microstructure Core
    microstructure_core = vol_adjusted_flow * volume_distribution
    
    # Divergence Core
    divergence_core = price_efficiency_div * volume_flow_div
    
    # Final Alpha
    alpha = (core_efficiency * microstructure_core * divergence_core * 
             intraday_efficiency_change * volume_flow_consistency)
    
    return alpha
