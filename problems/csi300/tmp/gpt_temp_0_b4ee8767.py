import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum Divergence with Volume-Amount Synchronization
    """
    data = df.copy()
    
    # Multi-period momentum calculation
    data['mom_3'] = data['close'] / data['close'].shift(3) - 1
    data['mom_8'] = data['close'] / data['close'].shift(8) - 1
    data['mom_21'] = data['close'] / data['close'].shift(21) - 1
    data['mom_55'] = data['close'] / data['close'].shift(55) - 1
    
    # Divergence detection
    momentum_returns = data[['mom_3', 'mom_8', 'mom_21', 'mom_55']]
    data['max_momentum_diff'] = momentum_returns.diff(axis=1).abs().max(axis=1)
    data['momentum_variance'] = momentum_returns.var(axis=1)
    
    # Volume dynamics
    data['volume_momentum'] = data['volume'].rolling(5).mean() / data['volume'].rolling(20).mean()
    data['volume_skew'] = data['volume'].rolling(20).apply(lambda x: x.skew(), raw=True)
    
    # Amount efficiency features
    data['amount_volume_ratio'] = data['amount'] / data['volume']
    data['amount_momentum'] = data['amount'].rolling(5).mean() / data['amount'].rolling(20).mean()
    
    # Synchronization metrics
    data['price_volume_corr'] = data['close'].rolling(10).corr(data['volume'])
    data['price_amount_corr'] = data['close'].rolling(10).corr(data['amount'])
    data['volume_amount_corr'] = data['volume'].rolling(10).corr(data['amount'])
    
    # Asymmetric Confirmation Signals
    data['volume_up_amount_down'] = ((data['volume'] > data['volume'].shift(1)) & 
                                   (data['amount'] < data['amount'].shift(1))).astype(int)
    data['volume_down_amount_up'] = ((data['volume'] < data['volume'].shift(1)) & 
                                   (data['amount'] > data['amount'].shift(1))).astype(int)
    
    # Breakout confirmation
    data['momentum_volume_conf'] = data['max_momentum_diff'] * data['volume_momentum']
    data['momentum_amount_conf'] = data['max_momentum_diff'] * data['amount_momentum']
    
    # Composite Alpha Generation
    # Multi-dimensional divergence score
    divergence_score = (data['max_momentum_diff'] * 0.4 + 
                       data['momentum_variance'] * 0.3 + 
                       (1 - data['volume_amount_corr'].abs()) * 0.3)
    
    # Synchronization-weighted momentum signals
    sync_weight = (data['price_volume_corr'].abs() + data['price_amount_corr'].abs()) / 2
    weighted_momentum = data['mom_21'] * sync_weight
    
    # Asymmetry-confirmed breakout patterns
    asymmetry_signal = (data['volume_up_amount_down'] - data['volume_down_amount_up']) * 0.2
    
    # Final composite alpha
    alpha = (divergence_score * 0.5 + 
             weighted_momentum * 0.3 + 
             data['momentum_volume_conf'] * 0.1 + 
             data['momentum_amount_conf'] * 0.1 + 
             asymmetry_signal)
    
    return alpha
