import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Quantum Entropic Momentum Alpha Factor
    Combines momentum, efficiency, volume dynamics, and asymmetry patterns using entropy-based transformations
    """
    data = df.copy()
    
    # Helper function for entropy calculation
    def entropy_term(x):
        abs_x = np.abs(x)
        # Avoid log(0) and division by zero
        safe_abs_x = np.where(abs_x == 0, 1e-10, abs_x)
        return -safe_abs_x * np.log(safe_abs_x)
    
    # Entropic Momentum Framework
    # Micro-Scale Momentum (1-day)
    micro_momentum = (data['close'] - data['close'].shift(1)) * \
                    entropy_term(data['close'] / data['close'].shift(1) - 1) * \
                    data['volume']
    
    # Meso-Scale Momentum (5-day)
    meso_momentum = (data['close'] - data['close'].shift(5)) * \
                   entropy_term(data['close'] / data['close'].shift(5) - 1) * \
                   (data['volume'] / data['volume'].shift(5))
    
    # Macro-Scale Momentum (21-day)
    macro_momentum = (data['close'] - data['close'].shift(21)) * \
                    entropy_term(data['close'] / data['close'].shift(21) - 1) * \
                    (data['high'] - data['low'])
    
    # Quantum Price Efficiency
    # Range Entropy
    range_ratio = (data['close'] - data['open']) / (data['high'] - data['low'])
    range_entropy = entropy_term(range_ratio) * data['volume']
    
    # Gap Entropy
    gap_ratio = (data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    gap_entropy = entropy_term(gap_ratio) * data['amount']
    
    # Volatility Entropy
    vol_ratio = (data['close'] - data['close'].shift(1)) / (data['close'].shift(1) - data['close'].shift(2))
    volatility_entropy = entropy_term(vol_ratio) * data['volume']
    
    # Volume Entropic Dynamics
    # Volume Change Entropy
    vol_change_entropy = entropy_term(data['volume'] / data['volume'].shift(1)) * \
                        (data['close'] - data['close'].shift(1))
    
    # Volume-Price Entropy
    vol_price_ratio = np.sign(data['close'] - data['close'].shift(1)) * data['volume'] / data['amount']
    vol_price_entropy = entropy_term(vol_price_ratio) * (data['high'] - data['low'])
    
    # Volume Persistence (5-day rolling)
    def volume_persistence_calc(vol_series):
        vol_ratio = vol_series / vol_series.shift(1)
        return np.sign(vol_series - vol_series.shift(1)) * entropy_term(vol_ratio)
    
    volume_persistence = data['volume'].rolling(window=5).apply(
        lambda x: volume_persistence_calc(pd.Series(x)).sum(), raw=False
    )
    
    # Quantum Asymmetry Patterns
    # Opening Asymmetry
    opening_ratio = (data['close'] - data['open']) / (data['open'] - data['close'].shift(1))
    high_low_ratio = (data['high'] - data['close']) / (data['close'] - data['low'])
    opening_asymmetry = entropy_term(opening_ratio) * high_low_ratio
    
    # Closing Asymmetry
    closing_ratio = (data['high'] - data['close']) / (data['close'] - data['low'])
    closing_asymmetry = entropy_term(closing_ratio) * np.abs(data['close'] - data['open']) * \
                       (data['volume'] / data['amount'])
    
    # Alpha Synthesis
    # Component weights
    momentum_component = (0.35 * micro_momentum + 
                         0.35 * meso_momentum + 
                         0.30 * macro_momentum)
    
    efficiency_component = (0.40 * range_entropy + 
                          0.35 * gap_entropy + 
                          0.25 * volatility_entropy)
    
    volume_component = (0.40 * vol_change_entropy + 
                      0.35 * vol_price_entropy + 
                      0.25 * volume_persistence)
    
    asymmetry_component = (0.60 * opening_asymmetry + 
                         0.40 * closing_asymmetry)
    
    # Final Alpha
    alpha = momentum_component * efficiency_component * volume_component * asymmetry_component
    
    # Handle any remaining NaN values
    alpha = alpha.fillna(0)
    
    return alpha
