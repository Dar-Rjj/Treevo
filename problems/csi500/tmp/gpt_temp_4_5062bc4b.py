import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adaptive Convergence Factor
    Combines momentum convergence, intraday efficiency, and volume confirmation
    with volatility regime adaptation for improved signal quality
    """
    data = df.copy()
    
    # Volatility Regime Identification
    data['price_volatility'] = data['close'].pct_change().rolling(window=20, min_periods=15).std()
    data['volume_volatility'] = data['volume'].pct_change().rolling(window=20, min_periods=15).std()
    
    # Normalize volatility measures
    data['price_vol_norm'] = (data['price_volatility'] - data['price_volatility'].rolling(window=60, min_periods=40).mean()) / data['price_volatility'].rolling(window=60, min_periods=40).std()
    data['volume_vol_norm'] = (data['volume_volatility'] - data['volume_volatility'].rolling(window=60, min_periods=40).mean()) / data['volume_volatility'].rolling(window=60, min_periods=40).std()
    
    # Volatility regime classification
    data['volatility_regime'] = np.where(
        data['price_vol_norm'] > 1.0, 
        'high', 
        np.where(data['price_vol_norm'] < -0.5, 'low', 'normal')
    )
    
    # Multi-Timeframe Momentum Analysis
    data['ultra_short_momentum'] = data['close'].pct_change(periods=3)
    data['short_momentum'] = data['close'].pct_change(periods=8)
    data['medium_momentum'] = data['close'].pct_change(periods=21)
    
    # Momentum convergence score
    data['momentum_convergence'] = (
        np.sign(data['ultra_short_momentum']) * np.sign(data['short_momentum']) * 
        np.sign(data['medium_momentum']) * 
        (abs(data['ultra_short_momentum']) + abs(data['short_momentum']) + abs(data['medium_momentum'])) / 3
    )
    
    # Intraday Efficiency Assessment
    data['range_efficiency'] = np.where(
        data['high'] != data['low'],
        (data['close'] - data['open']) / (data['high'] - data['low']),
        0
    )
    data['efficiency_persistence'] = data['range_efficiency'].rolling(window=3, min_periods=2).mean()
    data['price_distribution'] = np.where(
        data['high'] != data['low'],
        (data['close'] - data['low']) / (data['high'] - data['low']),
        0.5
    )
    
    # Volume Confirmation Patterns
    data['volume_trend'] = data['volume'].ewm(span=10, adjust=False).mean()
    data['volume_efficiency'] = data['volume_trend'] * abs(data['range_efficiency'])
    
    # Price-volume alignment (5-day rolling correlation)
    data['efficiency_rolling'] = data['range_efficiency'].rolling(window=5, min_periods=3).mean()
    data['volume_rolling'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['price_volume_alignment'] = data['efficiency_rolling'].rolling(window=10, min_periods=7).corr(data['volume_rolling'])
    
    # Regime-Specific Convergence Detection
    # High volatility convergence components
    data['extreme_momentum_reversal'] = np.where(
        data['volatility_regime'] == 'high',
        data['momentum_convergence'] * data['price_vol_norm'],
        0
    )
    
    # Gap analysis for high volatility
    data['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_recovery_efficiency'] = np.where(
        data['volatility_regime'] == 'high',
        data['range_efficiency'] * (1 - abs(data['gap'])),
        data['range_efficiency']
    )
    
    data['volume_confirmation'] = np.where(
        data['volatility_regime'] == 'high',
        data['volume_trend'] * data['efficiency_persistence'],
        data['volume_trend']
    )
    
    # Normal volatility convergence components
    data['momentum_efficiency_alignment'] = np.where(
        data['volatility_regime'] == 'normal',
        data['range_efficiency'] * data['momentum_convergence'],
        data['range_efficiency']
    )
    
    data['price_volume_convergence'] = np.where(
        data['volatility_regime'] == 'normal',
        data['price_volume_alignment'] * data['volume_trend'],
        data['price_volume_alignment']
    )
    
    data['microstructure_efficiency'] = np.where(
        data['volatility_regime'] == 'normal',
        data['range_efficiency'] * data['price_distribution'],
        data['range_efficiency']
    )
    
    # Factor Construction - Core convergence component
    # Regime-weighted momentum convergence
    data['regime_weighted_momentum'] = np.where(
        data['volatility_regime'] == 'high',
        data['extreme_momentum_reversal'] * 0.6 + data['momentum_convergence'] * 0.4,
        np.where(
            data['volatility_regime'] == 'normal',
            data['momentum_convergence'] * 0.7 + data['momentum_efficiency_alignment'] * 0.3,
            data['momentum_convergence'] * 0.8  # low volatility
        )
    )
    
    # Volume-confirmed efficiency persistence
    data['volume_confirmed_efficiency'] = (
        data['efficiency_persistence'] * data['volume_confirmation'] / 
        data['volume_confirmation'].rolling(window=20, min_periods=15).mean()
    )
    
    # Intraday pattern strength
    data['intraday_pattern_strength'] = (
        data['range_efficiency'] * data['price_distribution'] * 
        (1 - abs(data['gap']))
    )
    
    # Microstructure enhancement
    # Opening gap adjustment by volatility regime
    data['gap_adjustment'] = np.where(
        data['volatility_regime'] == 'high',
        1 - abs(data['gap']) * 0.8,
        np.where(
            data['volatility_regime'] == 'normal',
            1 - abs(data['gap']) * 0.5,
            1 - abs(data['gap']) * 0.3  # low volatility
        )
    )
    
    # Range efficiency context adaptation
    data['range_efficiency_adapted'] = data['range_efficiency'] * data['gap_adjustment']
    
    # Price distribution alignment
    data['price_distribution_aligned'] = np.where(
        data['price_distribution'] > 0.5,
        data['price_distribution'] * data['range_efficiency_adapted'],
        (1 - data['price_distribution']) * data['range_efficiency_adapted']
    )
    
    # Signal Validation - Noise reduction
    # Volatility-based extreme value filtering
    volatility_threshold = data['price_volatility'].rolling(window=60, min_periods=40).quantile(0.9)
    data['volatility_filter'] = np.where(data['price_volatility'] > volatility_threshold, 0.5, 1.0)
    
    # Regime-specific pattern strength thresholds
    data['pattern_strength'] = np.where(
        data['volatility_regime'] == 'high',
        data['intraday_pattern_strength'] * 0.7,
        np.where(
            data['volatility_regime'] == 'normal',
            data['intraday_pattern_strength'] * 0.9,
            data['intraday_pattern_strength'] * 1.1  # low volatility - amplify signals
        )
    )
    
    # Final factor construction
    data['core_convergence'] = (
        data['regime_weighted_momentum'] * 0.4 +
        data['volume_confirmed_efficiency'] * 0.3 +
        data['pattern_strength'] * 0.3
    ) * data['volatility_filter']
    
    # Multi-day convergence confirmation (3-day smoothing)
    data['convergence_confirmation'] = data['core_convergence'].rolling(window=3, min_periods=2).mean()
    
    # Efficiency trend stability (5-day trend)
    data['efficiency_trend'] = data['range_efficiency_adapted'].rolling(window=5, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else 0
    )
    
    # Volume pattern persistence
    data['volume_persistence'] = data['volume_confirmation'].rolling(window=5, min_periods=3).std()
    data['volume_stability'] = 1 / (1 + data['volume_persistence'])
    
    # Adaptive Output - Final factor with market context adaptation
    data['final_factor'] = (
        data['convergence_confirmation'] * 0.5 +
        data['efficiency_trend'] * 0.3 +
        data['volume_stability'] * 0.2
    )
    
    # Volatility regime sensitivity adjustment
    data['regime_sensitivity'] = np.where(
        data['volatility_regime'] == 'high',
        data['final_factor'] * 0.8,  # Reduce sensitivity in high volatility
        np.where(
            data['volatility_regime'] == 'normal',
            data['final_factor'] * 1.0,
            data['final_factor'] * 1.2  # Increase sensitivity in low volatility
        )
    )
    
    # Final normalization and cleaning
    factor = data['regime_sensitivity'].copy()
    factor = factor.replace([np.inf, -np.inf], np.nan)
    
    # Z-score normalization within volatility regime groups
    def normalize_by_regime(group):
        if len(group) > 10:
            return (group - group.mean()) / group.std()
        else:
            return group
    
    # Apply regime-specific normalization
    factor_normalized = factor.groupby(data['volatility_regime']).transform(normalize_by_regime)
    
    return factor_normalized
