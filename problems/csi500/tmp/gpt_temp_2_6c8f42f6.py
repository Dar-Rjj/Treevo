import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate returns for volatility
    data['returns'] = data['close'] / data['close'].shift(1) - 1
    
    # Dual-Regime Detection
    # Volatility Regime Analysis
    data['short_term_vol'] = data['returns'].rolling(window=10, min_periods=10).apply(
        lambda x: np.sqrt(np.nansum(x**2) / len(x)), raw=False
    )
    data['long_term_vol'] = data['returns'].rolling(window=20, min_periods=20).apply(
        lambda x: np.sqrt(np.nansum(x**2) / len(x)), raw=False
    )
    data['volatility_regime'] = np.sign(data['short_term_vol'] - data['long_term_vol'])
    
    # Liquidity Regime Analysis
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=5).mean()
    data['volume_concentration'] = data['volume'] / data['volume_ma_5']
    
    data['amount_ma_5'] = data['amount'].rolling(window=5, min_periods=5).mean()
    data['amount_stability'] = 1 / (1 + data['amount'].rolling(window=5, min_periods=5).apply(
        lambda x: np.sqrt(np.nansum((x - np.nanmean(x))**2) / len(x)), raw=False
    ))
    data['liquidity_regime'] = np.sign(data['volume_concentration'] - 1)
    
    # Breakout Detection with Acceleration
    # Breakout Strength Calculation
    data['high_20_max'] = data['high'].rolling(window=19, min_periods=19).max().shift(1)
    data['low_20_min'] = data['low'].rolling(window=19, min_periods=19).min().shift(1)
    data['high_breakout_20'] = data['high'] / data['high_20_max'] - 1
    data['low_breakout_20'] = 1 - data['low'] / data['low_20_min']
    data['net_breakout'] = data['high_breakout_20'] - data['low_breakout_20']
    
    # Momentum Acceleration Analysis
    data['momentum_5'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_10'] = data['close'].shift(5) / data['close'].shift(10) - 1
    data['momentum_acceleration'] = data['momentum_5'] - data['momentum_10']
    data['acceleration_weighted_breakout'] = data['net_breakout'] * (1 + np.abs(data['momentum_acceleration']))
    
    # Range-Based Confirmation
    # True Range Analysis
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['prev_close']),
            np.abs(data['low'] - data['prev_close'])
        )
    )
    data['range_efficiency'] = data['true_range'] / data['true_range'].rolling(window=5, min_periods=5).mean()
    data['range_regime'] = np.sign(data['range_efficiency'] - 1)
    
    # Elasticity Adjustment
    data['range_ma_3'] = data['true_range'].rolling(window=3, min_periods=3).mean()
    data['range_elasticity'] = (data['true_range'] / data['range_ma_3']) - 1
    data['range_confirmed_breakout'] = data['acceleration_weighted_breakout'] * (1 + data['range_elasticity'] * data['range_regime'])
    
    # Regime-Adaptive Signal Construction
    # Volatility-Weighted Breakout
    data['high_vol_adjustment'] = data['range_confirmed_breakout'] * (1 + np.abs(data['volatility_regime']))
    data['low_vol_adjustment'] = data['range_confirmed_breakout'] * (1 - np.abs(data['volatility_regime']))
    data['volatility_adaptive_breakout'] = np.where(
        data['volatility_regime'] > 0,
        data['high_vol_adjustment'],
        data['low_vol_adjustment']
    )
    
    # Liquidity-Enhanced Signal
    data['volume_regime_multiplier'] = 1 + (data['liquidity_regime'] * data['volume_concentration'])
    data['amount_stability_weight'] = data['volume_concentration'] * data['amount_stability']
    data['liquidity_adjusted_breakout'] = data['volatility_adaptive_breakout'] * data['volume_regime_multiplier'] * data['amount_stability_weight']
    
    # Final Alpha Generation
    # Regime Consistency Evaluation
    data['regime_alignment'] = data['volatility_regime'] * data['liquidity_regime'] * data['range_regime']
    data['strong_regime_signal'] = np.abs(data['regime_alignment']) > 0
    
    # Adaptive Factor Output
    data['strong_regime_factor'] = data['liquidity_adjusted_breakout'] * (1 + data['regime_alignment'])
    data['weak_regime_factor'] = data['liquidity_adjusted_breakout'] * 0.5
    
    data['final_alpha'] = np.where(
        data['strong_regime_signal'],
        data['strong_regime_factor'],
        data['weak_regime_factor']
    )
    
    return data['final_alpha']
