import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Fractal Momentum with Liquidity-Weighted Regime Switching
    """
    data = df.copy()
    
    # Fractal Momentum Structure
    # Multi-scale momentum calculation
    data['mom_3'] = data['close'] - data['close'].shift(3)
    data['mom_8'] = data['close'] - data['close'].shift(8)
    data['mom_13'] = data['close'] - data['close'].shift(13)
    
    # Momentum regime analysis
    data['mom_alignment'] = ((data['mom_3'] > 0) & (data['mom_8'] > 0) & (data['mom_13'] > 0)).astype(int) - \
                           ((data['mom_3'] < 0) & (data['mom_8'] < 0) & (data['mom_13'] < 0)).astype(int)
    
    # Momentum persistence (3-day same direction)
    data['mom_persistence'] = 0
    for i in range(3, len(data)):
        if (data['mom_3'].iloc[i] > 0 and data['mom_3'].iloc[i-1] > 0 and data['mom_3'].iloc[i-2] > 0):
            data.loc[data.index[i], 'mom_persistence'] = 1
        elif (data['mom_3'].iloc[i] < 0 and data['mom_3'].iloc[i-1] < 0 and data['mom_3'].iloc[i-2] < 0):
            data.loc[data.index[i], 'mom_persistence'] = -1
    
    # Regime transition detection
    data['regime_transition'] = ((data['mom_alignment'] != data['mom_alignment'].shift(1)) & 
                                (data['mom_alignment'].shift(1) != 0)).astype(int)
    
    # Liquidity Impact Assessment
    # Price range efficiency
    data['price_efficiency'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Relative volume weighting
    data['rel_volume'] = data['volume'] / (data['volume'].shift(1) + 1e-8)
    
    # Trade size dynamics
    data['avg_trade_size'] = data['amount'] / (data['volume'] + 1e-8)
    data['trade_size_vol'] = data['avg_trade_size'].rolling(window=5, min_periods=3).std()
    data['trade_size_impact'] = data['avg_trade_size'] / (data['trade_size_vol'] + 1e-8)
    
    # Volume-weighted price efficiency
    data['vw_price_efficiency'] = data['price_efficiency'] * np.log1p(data['rel_volume'])
    
    # Regime-Dependent Signal Processing
    # Price regime identification
    data['price_position'] = (data['close'] - data['low'].rolling(window=20, min_periods=10).min()) / \
                            (data['high'].rolling(window=20, min_periods=10).max() - 
                             data['low'].rolling(window=20, min_periods=10).min() + 1e-8)
    
    # Regime strength assessment
    data['regime_strength'] = (abs(data['mom_3']) + abs(data['mom_8']) + abs(data['mom_13'])) / 3
    
    # Signal transformation components
    # Regime-specific weighting
    data['regime_weight'] = np.where(data['mom_alignment'] != 0, 
                                   data['regime_strength'] * (1 + data['regime_transition'] * 0.5), 
                                   0.5)
    
    # Transition amplification
    data['transition_amp'] = np.where(data['regime_transition'] == 1, 1.5, 1.0)
    
    # Stable regime smoothing
    data['stability_smooth'] = np.where(data['mom_persistence'] != 0, 0.8, 1.0)
    
    # Alpha Synthesis
    # Multi-timeframe integration - Fractal momentum combination
    data['fractal_momentum'] = (data['mom_3'] * 0.4 + data['mom_8'] * 0.35 + data['mom_13'] * 0.25)
    
    # Liquidity factor integration
    data['liquidity_factor'] = (data['vw_price_efficiency'] * 0.6 + 
                               np.tanh(data['trade_size_impact']) * 0.4)
    
    # Final alpha generation
    # Regime-adaptive signal output
    base_signal = data['fractal_momentum'] * data['liquidity_factor']
    
    # Multi-factor confirmation with regime adaptation
    alpha = (base_signal * data['regime_weight'] * data['transition_amp'] * 
             data['stability_smooth'] * data['mom_alignment'])
    
    # Normalize the final alpha
    alpha_std = alpha.rolling(window=20, min_periods=10).std()
    alpha = alpha / (alpha_std + 1e-8)
    
    return alpha
