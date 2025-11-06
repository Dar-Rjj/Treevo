import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily returns
    data['returns'] = data['close'] / data['close'].shift(1) - 1
    
    # Volatility Asymmetry Detection
    def calc_asymmetry_volatility(window_returns):
        positive_returns = window_returns[window_returns > 0]
        negative_returns = window_returns[window_returns < 0]
        
        pos_vol = positive_returns.std() if len(positive_returns) > 1 else 0
        neg_vol = negative_returns.std() if len(negative_returns) > 1 else 0
        
        # Avoid division by zero
        if neg_vol == 0:
            return 1.0
        return pos_vol / neg_vol
    
    # Calculate rolling volatility asymmetry
    volatility_asymmetry = []
    for i in range(len(data)):
        if i < 4:
            volatility_asymmetry.append(1.0)
        else:
            window_returns = data['returns'].iloc[i-4:i+1]
            vol_ratio = calc_asymmetry_volatility(window_returns)
            volatility_asymmetry.append(vol_ratio)
    
    data['volatility_asymmetry_ratio'] = volatility_asymmetry
    
    # Microstructural Anchoring Components
    data['price_anchoring'] = ((data['high'] + data['low']) / 2) - ((data['high'].shift(1) + data['low'].shift(1)) / 2)
    
    # Volume anchoring with condition
    price_change_threshold = 0.005
    small_price_change = (abs(data['close'] - data['close'].shift(1)) / data['close'].shift(1)) < price_change_threshold
    data['volume_anchoring'] = np.where(small_price_change, 
                                       data['volume'] / data['volume'].shift(1), 
                                       1.0)
    
    data['amount_anchoring'] = data['amount'] / data['amount'].shift(1)
    
    # Regime-Adaptive Signal Construction
    data['upside_vol_signal'] = data['volatility_asymmetry_ratio'] * data['price_anchoring']
    data['downside_vol_signal'] = (1 / data['volatility_asymmetry_ratio']) * data['volume_anchoring']
    data['balanced_vol_signal'] = data['amount_anchoring'] * data['price_anchoring']
    
    # Anchor Decay Dynamics
    data['price_anchor_decay'] = abs(data['close'] - data['close'].shift(1)) / abs(data['close'].shift(1) - data['close'].shift(2))
    data['volume_anchor_decay'] = (data['volume'] / data['volume'].shift(1)) - (data['volume'].shift(1) / data['volume'].shift(2))
    data['amount_anchor_decay'] = (data['amount'] / data['amount'].shift(1)) - (data['amount'].shift(1) / data['amount'].shift(2))
    
    # Replace infinities and handle division by zero
    data['price_anchor_decay'] = data['price_anchor_decay'].replace([np.inf, -np.inf], 1.0)
    data['price_anchor_decay'] = data['price_anchor_decay'].fillna(1.0)
    
    # Final Alpha Synthesis
    # Use balanced volatility signal as regime adaptive signal
    data['core_component'] = data['balanced_vol_signal'] * (1 + data['price_anchor_decay'])
    data['confirmation_multiplier'] = data['volume_anchor_decay'] * data['amount_anchor_decay']
    
    # Final alpha factor
    alpha = data['core_component'] * data['confirmation_multiplier']
    
    # Clean extreme values
    alpha = alpha.replace([np.inf, -np.inf], np.nan)
    alpha = alpha.fillna(0)
    
    return alpha
