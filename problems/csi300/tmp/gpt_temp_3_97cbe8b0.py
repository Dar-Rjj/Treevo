import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum-Volume Efficiency Gap with Regime-Aware Signal Enhancement
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Assessment Momentum Framework
    # Price Momentum Efficiency Analysis
    data['price_momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_efficiency'] = data['price_momentum_5d'] / (data['high'] - data['low'])
    data['momentum_persistence'] = np.sign(data['price_momentum_5d']) * np.sign(data['price_momentum_5d'].shift(2))
    
    # Volume Efficiency Assessment
    data['volume_intensity'] = data['volume'] / (data['high'] - data['low'])
    data['volume_efficiency_ratio'] = data['volume_intensity'] / data['volume_intensity'].shift(5)
    data['volume_momentum_gap'] = data['volume_efficiency_ratio'] - data['price_momentum_5d']
    
    # Efficiency Divergence Detection
    data['efficiency_gap_magnitude'] = abs(data['volume_momentum_gap'])
    data['efficiency_gap_direction'] = np.sign(data['volume_momentum_gap'])
    
    # Market Regime Context Integration
    # Price Regime Classification
    data['high_10d'] = data['high'].rolling(window=10, min_periods=1).max()
    data['low_10d'] = data['low'].rolling(window=10, min_periods=1).min()
    data['price_range_10d'] = data['high_10d'] - data['low_10d']
    data['regime_intensity'] = (data['close'] - data['low_10d']) / data['price_range_10d']
    
    # Volume Regime Analysis
    data['volume_percentile'] = data['volume'].rolling(window=10, min_periods=1).apply(
        lambda x: (x.rank(pct=True).iloc[-1] * 100) if len(x) == 10 else np.nan
    )
    data['volume_regime_strength'] = data['volume_percentile'] / 100
    data['volume_breakout'] = (data['volume_percentile'] > 80).astype(int)
    
    # Gap-Based Signal Generation
    # Bullish Efficiency Gap Conditions
    bullish_condition = (data['price_momentum_5d'] > 0) & (data['volume_momentum_gap'] < 0)
    data['bullish_signal'] = np.where(
        bullish_condition,
        data['volume_efficiency_ratio'] * (1 - abs(data['price_momentum_5d'])) * data['regime_intensity'],
        0
    )
    
    # Bearish Efficiency Gap Conditions
    bearish_condition = (data['price_momentum_5d'] < 0) & (data['volume_momentum_gap'] > 0)
    data['bearish_signal'] = np.where(
        bearish_condition,
        -data['volume_efficiency_ratio'] * abs(data['price_momentum_5d']) * (1 - data['regime_intensity']),
        0
    )
    
    # Signal Quality Assessment
    data['signal_quality'] = data['efficiency_gap_magnitude'] * data['volume_regime_strength']
    
    # Multi-Dimensional Validation Framework
    # Regime Context Validation
    trending_regime = (data['regime_intensity'] > 0.7) | (data['regime_intensity'] < 0.3)
    ranging_regime = (data['regime_intensity'] >= 0.3) & (data['regime_intensity'] <= 0.7)
    
    # Apply regime-specific scaling
    regime_scaling = np.where(
        trending_regime, 1.2,  # Boost signals in trending regimes
        np.where(ranging_regime, 0.8, 1.0)  # Reduce signals in ranging regimes
    )
    
    # Volume Confirmation Enhancement
    volume_confirmation = data['volume_breakout'] * data['volume_regime_strength']
    
    # Composite Alpha Generation
    # Regime-Aware Efficiency Gap Factor
    base_signal = data['bullish_signal'] + data['bearish_signal']
    
    # Multi-Dimensional Predictive Signal
    data['composite_alpha'] = (
        base_signal * 
        regime_scaling * 
        data['signal_quality'] * 
        (1 + volume_confirmation * 0.3)  # Volume confirmation boost
    )
    
    # Final Alpha Output with additional smoothing
    alpha = data['composite_alpha'].rolling(window=3, min_periods=1).mean()
    
    return alpha
