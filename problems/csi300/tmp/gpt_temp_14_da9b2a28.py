import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Add small epsilon to avoid division by zero
    eps = 0.001
    
    # Fractal Price Efficiency Components
    # Fractal Momentum Efficiency
    data['fractal_momentum_eff'] = (
        (data['close'] - data['close'].shift(1)) / 
        (data['close'].shift(1) - data['close'].shift(2) + eps) * 
        (data['close'] - data['close'].shift(3)) / 
        (np.abs(data['close'] - data['close'].shift(3)) + eps) * 
        (data['close'] - data['open']) / 
        (data['high'] - data['low'] + eps)
    )
    
    # Range Efficiency Momentum
    data['range_eff_momentum'] = (
        (data['close'] - data['close'].shift(3)) / 
        (data['high'] - data['low'] + eps) * 
        (data['close'] - data['open']) / 
        (np.abs(data['open'] - data['close'].shift(1)) + eps)
    )
    
    # Volatility Expansion Efficiency
    data['vol_expansion_eff'] = (
        ((data['high'] - data['low']) / (data['high'].shift(3) - data['low'].shift(3) + eps) - 1) * 
        (data['close'] - data['close'].shift(1)) / 
        (data['high'] - data['low'] + eps) * 
        (data['close'] - data['open']) / 
        (data['high'] - data['low'] + eps)
    )
    
    # Microstructure Liquidity Integration Components
    # Volatility-Adjusted Volume Flow
    data['vol_adj_volume_flow'] = (
        (data['close'] - data['low']) / 
        (data['high'] - data['low'] + eps) * 
        (data['volume'] - data['volume'].shift(1)) / 
        (data['volume'] + data['volume'].shift(1) + eps) * 
        np.sign(data['close'] - data['close'].shift(1)) * 
        np.sign(data['volume'] - data['volume'].shift(1))
    )
    
    # Efficiency Liquidity Momentum
    data['eff_liquidity_momentum'] = (
        (data['volume'] - data['volume'].shift(3)) / 
        (data['volume'].shift(3) + eps) * 
        (data['close'] - data['close'].shift(3)) / 
        (data['close'].shift(3) + eps) * 
        (data['close'] - data['open']) / 
        (data['high'] - data['low'] + eps)
    )
    
    # Volume Distribution Efficiency
    data['volume_dist_eff'] = (
        (data['volume'] * (data['open'] - data['low']) / (data['high'] - data['low'] + eps) - 
         data['volume'] * (data['high'] - data['close']) / (data['high'] - data['low'] + eps)) * 
        np.sign(data['close'] - data['close'].shift(1))
    )
    
    # Regime-Adaptive Divergence Components
    # Price-Efficiency Divergence
    data['price_eff_divergence'] = (
        (data['close'] - data['close'].shift(2)) / (data['close'].shift(2) + eps) - 
        (data['close'] - data['open']) / (data['high'] - data['low'] + eps)
    )
    
    # Volume-Flow Divergence
    data['volume_flow_divergence'] = (
        np.sign(data['volume'] - data['volume'].shift(1)) * np.sign(data['close'] - data['close'].shift(1)) - 
        np.sign(data['volume'].shift(2) - data['volume'].shift(3)) * np.sign(data['close'].shift(2) - data['close'].shift(3))
    )
    
    # Regime Divergence Signal
    data['regime_divergence_signal'] = (
        np.sign(data['price_eff_divergence']) * 
        np.sign(data['volume_flow_divergence']) * 
        (data['high'] - data['low']) / (data['high'].shift(3) - data['low'].shift(3) + eps)
    )
    
    # Multi-Timeframe Efficiency Components
    # Intraday Efficiency Change
    data['intraday_eff_change'] = (
        (data['close'] - data['open']) / (data['high'] - data['low'] + eps) - 
        (data['close'].shift(1) - data['open'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + eps)
    )
    
    # Volume Flow Change
    data['volume_flow_change'] = (
        np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['volume'] - data['volume'].shift(1)) - 
        np.sign(data['close'].shift(1) - data['close'].shift(2)) * np.sign(data['volume'].shift(1) - data['volume'].shift(2))
    )
    
    # Efficiency Consistency (rolling count of same sign)
    intraday_eff_sign = np.sign((data['close'] - data['open']) / (data['high'] - data['low'] + eps))
    data['eff_consistency'] = intraday_eff_sign.rolling(window=4).apply(
        lambda x: np.sum(x == x.iloc[-1]) if len(x) == 4 else np.nan, raw=False
    )
    
    # Volume Flow Consistency (rolling count of same sign)
    volume_flow_sign = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['volume'] - data['volume'].shift(1))
    data['volume_flow_consistency'] = volume_flow_sign.rolling(window=4).apply(
        lambda x: np.sum(x == x.iloc[-1]) if len(x) == 4 else np.nan, raw=False
    )
    
    # Composite Alpha Construction
    # Core Efficiency Factor
    data['fractal_eff_core'] = data['fractal_momentum_eff'] * data['range_eff_momentum'] * data['vol_expansion_eff']
    data['microstructure_core'] = data['vol_adj_volume_flow'] * data['eff_liquidity_momentum'] * data['volume_dist_eff']
    data['divergence_core'] = data['price_eff_divergence'] * data['volume_flow_divergence'] * data['regime_divergence_signal']
    
    # Multi-Timeframe Enhancement
    data['short_term_component'] = data['intraday_eff_change'] * data['volume_flow_change']
    data['medium_term_component'] = data['eff_consistency'] * data['volume_flow_consistency']
    
    # Final Alpha
    data['regime_adaptive_alpha'] = data['fractal_eff_core'] * data['microstructure_core'] * data['divergence_core']
    data['multi_timeframe_alpha'] = data['regime_adaptive_alpha'] * data['short_term_component'] * data['medium_term_component']
    
    # Return the final alpha factor
    return data['multi_timeframe_alpha']
