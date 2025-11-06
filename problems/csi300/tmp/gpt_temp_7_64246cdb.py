import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Regime Adaptive Momentum with Volume Confirmation
    """
    data = df.copy()
    
    # Dynamic Momentum Framework
    # Multi-timeframe Return Calculation
    data['R1'] = data['close'] / data['close'].shift(1) - 1
    data['R2'] = data['close'] / data['close'].shift(2) - 1
    data['R3'] = data['close'] / data['close'].shift(3) - 1
    data['R5'] = data['close'] / data['close'].shift(5) - 1
    data['R10'] = data['close'] / data['close'].shift(10) - 1
    data['R20'] = data['close'] / data['close'].shift(20) - 1
    
    # Volume Regime Classification
    # Volume Surge Detection
    data['volume_median_20'] = data['volume'].rolling(window=20).median()
    data['volume_ratio'] = data['volume'] / data['volume_median_20'].shift(1)
    data['volume_percentile'] = data['volume'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] > x).mean(), raw=False
    )
    data['volume_surge'] = ((data['volume_ratio'] > 2.0) | (data['volume_percentile'] > 0.9)).astype(int)
    
    # Volume Trend Analysis
    data['volume_momentum'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_acceleration'] = (data['volume']/data['volume'].shift(1)) / (data['volume'].shift(1)/data['volume'].shift(2))
    
    # Volume-Price Divergence Signals
    data['price_momentum_5'] = data['close'] / data['close'].shift(5) - 1
    data['bullish_divergence'] = ((data['price_momentum_5'] < 0) & 
                                 (data['volume_momentum'] > 0) & 
                                 (data['volume_surge'] == 1)).astype(int)
    data['bearish_divergence'] = ((data['price_momentum_5'] > 0) & 
                                 (data['volume_momentum'] < 0)).astype(int)
    
    # Volatility Regime Framework
    # Multi-scale Volatility Measurement
    data['intraday_vol'] = (data['high'] - data['low']) / data['close']
    data['close_vol_5'] = data['R1'].rolling(window=5).std()
    data['range_vol_5'] = data['intraday_vol'].rolling(window=5).mean()
    data['volatility_ratio'] = data['close_vol_5'] / data['close_vol_5'].rolling(window=20).mean()
    
    # Regime Classification
    data['high_vol_regime'] = (data['volatility_ratio'] > 1.2).astype(int)
    data['low_vol_regime'] = (data['volatility_ratio'] < 0.8).astype(int)
    data['transition_regime'] = ((data['volatility_ratio'] >= 0.8) & 
                                (data['volatility_ratio'] <= 1.2)).astype(int)
    
    # Price Pattern Integration
    # Support/Resistance Detection
    data['resistance_20'] = data['high'].rolling(window=20).max()
    data['support_20'] = data['low'].rolling(window=20).min()
    data['breakout'] = (data['close'] > data['resistance_20'].shift(1)).astype(int)
    data['breakdown'] = (data['close'] < data['support_20'].shift(1)).astype(int)
    
    # Gap Analysis
    data['overnight_gap'] = data['open'] / data['close'].shift(1) - 1
    
    # Trend Strength Assessment
    data['trend_alignment'] = (
        (np.sign(data['R5']) + np.sign(data['R10']) + np.sign(data['R20'])) / 3
    )
    
    # Adaptive Factor Combination
    # Regime-Dependent Signal Weighting
    # High volatility regime weights
    high_vol_momentum = (
        0.4 * data['R1'] + 
        0.3 * data['R2'] + 
        0.2 * data['R3'] + 
        0.1 * data['R5']
    )
    
    # Low volatility regime weights
    low_vol_momentum = (
        0.2 * data['R5'] + 
        0.3 * data['R10'] + 
        0.4 * data['R20'] + 
        0.1 * data['R3']
    )
    
    # Transition regime weights
    transition_momentum = (
        0.25 * data['R3'] + 
        0.25 * data['R5'] + 
        0.25 * data['R10'] + 
        0.25 * data['R20']
    )
    
    # Combine momentum based on regime
    regime_momentum = (
        data['high_vol_regime'] * high_vol_momentum +
        data['low_vol_regime'] * low_vol_momentum +
        data['transition_regime'] * transition_momentum
    )
    
    # Volume confirmation adjustment
    volume_confirmation = (
        1.0 + 
        0.3 * data['volume_surge'] * np.sign(regime_momentum) +
        0.2 * data['bullish_divergence'] -
        0.2 * data['bearish_divergence']
    )
    
    # Divergence Overlay System
    divergence_score = (
        0.5 * data['bullish_divergence'] -
        0.5 * data['bearish_divergence'] +
        0.3 * data['volume_momentum'] * np.sign(regime_momentum)
    )
    
    # Price pattern integration
    pattern_score = (
        0.2 * data['breakout'] -
        0.2 * data['breakdown'] +
        0.1 * data['trend_alignment'] +
        0.1 * np.sign(data['overnight_gap']) * abs(data['overnight_gap'])
    )
    
    # Dynamic Risk Adjustment
    volatility_scaling = 1.0 / (1.0 + data['close_vol_5'])
    
    # Final factor combination
    factor = (
        regime_momentum * 
        volume_confirmation * 
        (1.0 + 0.3 * divergence_score) * 
        (1.0 + 0.2 * pattern_score) * 
        volatility_scaling
    )
    
    return factor
