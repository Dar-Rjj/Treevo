import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Adaptive Asymmetric Liquidity Decay Factor
    """
    data = df.copy()
    
    # Volatility Classification
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    
    # Volatility regime based on 20-day percentiles
    data['vol_percentile'] = data['true_range'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
    )
    
    # Regime assignment
    data['vol_regime'] = 'transition'
    data.loc[data['vol_percentile'] > 0.7, 'vol_regime'] = 'high'
    data.loc[data['vol_percentile'] < 0.3, 'vol_regime'] = 'low'
    
    # Asymmetric Liquidity Components
    data['upside_liquidity'] = ((data['high'] - data['open']) * data['volume'] / 
                               np.maximum(data['open'] - data['low'], 0.001))
    data['downside_liquidity'] = ((data['open'] - data['low']) * data['volume'] / 
                                 np.maximum(data['high'] - data['open'], 0.001))
    data['asymmetric_ratio'] = data['upside_liquidity'] / np.maximum(data['downside_liquidity'], 0.001)
    
    # Regime-specific liquidity patterns
    # High volatility regime signals
    high_vol_mask = data['vol_regime'] == 'high'
    data['liquidity_fragility'] = data['asymmetric_ratio'] / np.maximum(data['volume'], 1)
    data['volume_price_divergence'] = ((data['high'] - data['low']) * 
                                     (data['volume'] / data['volume'].rolling(window=5).mean()))
    
    # Low volatility regime signals  
    low_vol_mask = data['vol_regime'] == 'low'
    min_volume_5d = data['volume'].rolling(window=5, min_periods=3).min()
    data['liquidity_accumulation'] = (data['volume'] / np.maximum(min_volume_5d, 1)) / np.maximum(data['high'] - data['low'], 0.001)
    data['volume_breakout_prob'] = ((data['close'] - data['open']) * data['volume'] / 
                                   np.maximum(data['high'] - data['low'], 0.001))
    
    # Transition regime signals
    trans_mask = data['vol_regime'] == 'transition'
    data['early_regime_change'] = ((data['asymmetric_ratio'] / data['asymmetric_ratio'].shift(3)) * 
                                 (data['volume'] / data['volume'].shift(3)))
    
    # Multi-scale liquidity decay dynamics
    data['short_term_fractal'] = data['asymmetric_ratio'] / data['asymmetric_ratio'].rolling(window=4, min_periods=2).mean()
    data['medium_term_fractal'] = data['asymmetric_ratio'] / data['asymmetric_ratio'].rolling(window=10, min_periods=5).max()
    data['long_term_consistency'] = data['asymmetric_ratio'] / data['asymmetric_ratio'].rolling(window=20, min_periods=10).min()
    
    # Price-liquidity decay correlation
    data['short_term_decay'] = ((data['close'] - data['close'].shift(1)) / 
                              np.maximum(data['asymmetric_ratio'] / data['asymmetric_ratio'].shift(1), 0.001))
    data['medium_term_decay'] = ((data['close'] - data['close'].shift(5)) / 
                               np.maximum(data['asymmetric_ratio'] / data['asymmetric_ratio'].rolling(window=4, min_periods=2).mean(), 0.001))
    
    # Liquidity decay momentum
    data['liquidity_acceleration'] = ((data['asymmetric_ratio'] / data['asymmetric_ratio'].shift(1)) / 
                                    np.maximum(data['asymmetric_ratio'].shift(1) / data['asymmetric_ratio'].shift(2), 0.001))
    
    # Microstructure liquidity decay analysis
    data['morning_session_decay'] = ((data['high'] - data['open']) * data['volume'] / 
                                   np.maximum(data['open'] - data['low'], 0.001))
    data['afternoon_session_decay'] = ((data['close'] - data['low']) * data['volume'] / 
                                     np.maximum(data['high'] - data['close'], 0.001))
    data['session_decay_spread'] = data['morning_session_decay'] - data['afternoon_session_decay']
    
    # Gap liquidity dynamics
    data['gap_micro_decay'] = ((data['open'] - data['close'].shift(1)) * data['volume'] / 
                             np.maximum(data['high'].shift(1) - data['low'].shift(1), 0.001))
    data['gap_absorption_decay'] = ((data['close'] - np.minimum(data['open'], data['close'].shift(1))) * data['volume'] / 
                                  np.maximum(abs(data['open'] - data['close'].shift(1)), 0.001))
    
    # Price level microstructure
    data['high_side_decay'] = ((data['high'] - data['close']) * data['volume'] / 
                             np.maximum(data['high'] - data['low'], 0.001))
    data['low_side_decay'] = ((data['close'] - data['low']) * data['volume'] / 
                            np.maximum(data['high'] - data['low'], 0.001))
    data['microstructure_imbalance'] = data['high_side_decay'] - data['low_side_decay']
    
    # Multi-timeframe decay convergence
    data['volume_trend_decay'] = data['volume'].rolling(window=3, min_periods=2).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / np.maximum(x.iloc[0], 1)
    )
    data['volatility_trend_decay'] = data['true_range'].rolling(window=3, min_periods=2).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / np.maximum(x.iloc[0], 0.001)
    )
    
    # Signal strength validation
    data['volume_deviation'] = data['volume'] / data['volume'].rolling(window=20, min_periods=10).median()
    data['volatility_ratio'] = data['true_range'] / data['true_range'].rolling(window=20, min_periods=10).median()
    
    # Composite alpha generation with regime weighting
    bullish_signals = pd.Series(0.0, index=data.index)
    bearish_signals = pd.Series(0.0, index=data.index)
    
    # Bullish signals
    bullish_signals += (low_vol_mask * data['liquidity_accumulation'] * 0.3)
    bullish_signals += (data['volume_breakout_prob'] * (data['volatility_trend_decay'] > 0) * 0.2)
    bullish_signals += (data['gap_absorption_decay'] * (data['gap_absorption_decay'] > 0) * 0.15)
    bullish_signals += (data['session_decay_spread'] * (data['session_decay_spread'] > 0) * 0.15)
    bullish_signals += (data['microstructure_imbalance'] * (data['microstructure_imbalance'] > 0) * 0.1)
    bullish_signals += (data['liquidity_acceleration'] * (data['liquidity_acceleration'] > 1) * 0.1)
    
    # Bearish signals
    bearish_signals += (high_vol_mask * data['liquidity_fragility'] * 0.3)
    bearish_signals += (data['volume_price_divergence'] * (data['volume_price_divergence'] < 0) * 0.2)
    bearish_signals += (data['gap_absorption_decay'] * (data['gap_absorption_decay'] < 0) * 0.15)
    bearish_signals += (data['session_decay_spread'] * (data['session_decay_spread'] < 0) * 0.15)
    bearish_signals += (data['microstructure_imbalance'] * (data['microstructure_imbalance'] < 0) * 0.1)
    bearish_signals += (data['liquidity_acceleration'] * (data['liquidity_acceleration'] < 1) * 0.1)
    
    # Regime-weighted final factor
    regime_weights = pd.Series(1.0, index=data.index)
    regime_weights[high_vol_mask] = 1.2  # Higher weight for high volatility regime
    regime_weights[low_vol_mask] = 0.8   # Lower weight for low volatility regime
    
    # Final composite factor
    composite_factor = (bullish_signals - bearish_signals) * regime_weights
    
    # Apply multi-scale validation
    short_term_align = (data['short_term_decay'] * data['short_term_fractal'] > 0).astype(float)
    medium_term_align = (data['medium_term_decay'] * data['medium_term_fractal'] > 0).astype(float)
    
    validation_score = (short_term_align + medium_term_align) / 2
    
    final_factor = composite_factor * validation_score
    
    return final_factor
