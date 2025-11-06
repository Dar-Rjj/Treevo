import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Directional Range Momentum
    data['current_range'] = data['high'] - data['low']
    data['prev_range'] = data['current_range'].shift(1)
    data['range_expansion'] = data['current_range'] / data['prev_range']
    
    # Net Range Bias
    data['upside_momentum'] = np.where(data['close'] > data['close'].shift(1), 
                                     data['high'] - data['high'].shift(1), 0)
    data['downside_momentum'] = np.where(data['close'] < data['close'].shift(1), 
                                       data['low'].shift(1) - data['low'], 0)
    data['net_range_bias'] = data['upside_momentum'] - data['downside_momentum']
    
    # Volatility Asymmetry Assessment
    data['upside_vol'] = data['high'] - data['close'].shift(1)
    data['downside_vol'] = data['close'].shift(1) - data['low']
    data['volatility_skew'] = data['upside_vol'] / data['downside_vol']
    
    # Asymmetric Range Adjustment
    data['vol_adjusted_range_expansion'] = data['range_expansion'] * data['volatility_skew']
    data['vol_adjusted_net_bias'] = data['net_range_bias'] * data['volatility_skew']
    
    # Liquidity-Volume Dynamics
    data['effective_spread'] = (data['amount'] / data['volume']) - data['close']
    data['volume_momentum'] = (data['volume'] / data['volume'].shift(5)) - 1
    data['price_volume_efficiency'] = data['net_range_bias'] / data['volume']
    
    # Liquidity Confirmation Signal
    data['liquidity_signal'] = np.where(
        (data['range_expansion'] > 1) & (data['volume_momentum'] > 0) & 
        (abs(data['effective_spread']) < data['effective_spread'].rolling(20).mean()),
        1, 0
    )
    
    # Adaptive Multi-scale Signal
    # Volatility-Adjusted Range Momentum
    data['immediate_momentum'] = data['vol_adjusted_range_expansion']
    data['short_term_momentum'] = data['net_range_bias'].rolling(5).mean()
    data['medium_term_momentum'] = data['net_range_bias'].rolling(15).apply(
        lambda x: 1 if (x > 0).sum() > (len(x) * 0.6) else -1 if (x < 0).sum() > (len(x) * 0.6) else 0
    )
    
    # Combined volatility-adjusted momentum
    data['combined_vol_momentum'] = (
        data['immediate_momentum'] * 0.4 + 
        data['short_term_momentum'] * 0.35 + 
        data['medium_term_momentum'] * 0.25
    )
    
    # Liquidity-Volume Multiplier
    data['price_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['liquidity_multiplier'] = np.where(
        (data['price_momentum'] < 0) & (data['volume_momentum'] > 0), 1.5,
        np.where((data['price_momentum'] > 0) & (data['volume_momentum'] < 0), -1.5, 1.0)
    )
    
    # Signal Convergence
    data['final_signal'] = (
        data['combined_vol_momentum'] * 
        data['liquidity_multiplier'] * 
        data['liquidity_signal']
    )
    
    # Apply final smoothing and normalization
    factor = data['final_signal'].rolling(3).mean()
    factor = (factor - factor.rolling(50).mean()) / factor.rolling(50).std()
    
    return factor
