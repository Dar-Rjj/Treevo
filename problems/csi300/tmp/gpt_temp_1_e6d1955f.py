import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Price Momentum with Volume-Pressure Dynamics
    """
    data = df.copy()
    
    # Multi-Timeframe Momentum Structure
    data['ultra_short_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['short_term_momentum'] = data['close'] / data['close'].shift(3) - 1
    data['medium_term_momentum'] = data['close'] / data['close'].shift(8) - 1
    data['momentum_curvature'] = (data['short_term_momentum'] - data['medium_term_momentum']) / np.abs(data['medium_term_momentum'].replace(0, np.nan))
    
    # Volume-Pressure Analysis
    data['volume_intensity'] = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    data['volume_momentum'] = data['volume'] / data['volume'].shift(3).replace(0, np.nan)
    data['volume_acceleration'] = (data['volume'] / data['volume'].shift(1)) / (data['volume'].shift(1) / data['volume'].shift(2)).replace(0, np.nan)
    data['volume_pressure_ratio'] = data['volume'] / (data['close'] - data['open']).replace(0, np.nan)
    
    # Price Efficiency Patterns
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['closing_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['opening_efficiency'] = (data['high'] - data['open']) / np.abs(data['open'] - data['close'].shift(1)).replace(0, np.nan)
    data['efficiency_persistence'] = data['intraday_efficiency'] / data['intraday_efficiency'].shift(3).replace(0, np.nan)
    
    # Dynamic Regime Classification
    # Volatility regime analysis
    data['returns'] = data['close'].pct_change()
    data['volatility_10d'] = data['returns'].rolling(window=10, min_periods=5).std()
    data['volatility_momentum'] = data['volatility_10d'] / data['volatility_10d'].shift(5).replace(0, np.nan)
    
    # Volume-pressure regimes
    data['volume_pressure_corr_5d'] = data['volume_intensity'].rolling(window=5, min_periods=3).corr(data['volume_pressure_ratio'])
    data['pressure_regime_persistence'] = data['volume_pressure_ratio'].rolling(window=5, min_periods=3).apply(lambda x: len(set(pd.cut(x, bins=2, labels=['Low', 'High']))) == 1)
    
    # Combined regime matrix
    data['volatility_regime'] = pd.cut(data['volatility_10d'], bins=2, labels=['Low', 'High'])
    data['pressure_regime'] = pd.cut(data['volume_pressure_ratio'], bins=2, labels=['Low', 'High'])
    
    # Gap-Driven Momentum Analysis
    data['gap_magnitude'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_extreme'] = data['gap_magnitude'].abs() > data['gap_magnitude'].rolling(window=15, min_periods=10).std() * 2
    data['intraday_up_momentum'] = (data['high'] - data['open']) / data['open']
    data['intraday_down_momentum'] = (data['open'] - data['low']) / data['open']
    data['range_utilization'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['gap_persistence'] = data['gap_magnitude'].rolling(window=3, min_periods=2).mean()
    data['gap_volume_ratio'] = data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean()
    
    # Regime-Adaptive Signal Processing
    # Volatility-context adjustments
    data['volatility_scaled_momentum'] = data['short_term_momentum'] / data['volatility_10d'].replace(0, np.nan)
    
    # Volume-pressure signal weighting
    high_pressure_mask = data['pressure_regime'] == 'High'
    low_pressure_mask = data['pressure_regime'] == 'Low'
    
    data['pressure_weighted_momentum'] = np.where(
        high_pressure_mask,
        data['short_term_momentum'] * data['volume_momentum'],
        np.where(
            low_pressure_mask,
            data['short_term_momentum'],
            data['short_term_momentum'] * (1 + data['volume_momentum']) / 2
        )
    )
    
    # Multi-timeframe integration
    data['multi_timeframe_score'] = (
        data['ultra_short_momentum'] * 0.2 +
        data['short_term_momentum'] * 0.4 +
        data['medium_term_momentum'] * 0.3 +
        data['momentum_curvature'] * 0.1
    )
    
    # Composite Alpha Generation
    # Core momentum-pressure integration
    data['momentum_pressure_core'] = (
        data['multi_timeframe_score'] * data['volume_pressure_ratio'] *
        data['efficiency_persistence']
    )
    
    # Gap sustainability factors
    data['gap_sustainability'] = (
        data['gap_magnitude'] * data['range_utilization'] *
        data['gap_volume_ratio'] * (1 - data['gap_extreme'].astype(float))
    )
    
    # Dynamic regime weighting
    volatility_weight = np.where(data['volatility_regime'] == 'High', 0.7, 1.3)
    pressure_weight = np.where(data['pressure_regime'] == 'High', 1.2, 0.8)
    regime_stability_weight = data['pressure_regime_persistence'].astype(float) * 1.5 + 0.5
    
    # Final predictive scoring
    alpha = (
        data['momentum_pressure_core'] * 0.5 +
        data['volatility_scaled_momentum'] * 0.2 +
        data['pressure_weighted_momentum'] * 0.15 +
        data['gap_sustainability'] * 0.15
    ) * volatility_weight * pressure_weight * regime_stability_weight
    
    # Cross-validation mechanisms
    momentum_pressure_alignment = np.sign(data['short_term_momentum']) == np.sign(data['volume_pressure_ratio'])
    efficiency_verification = data['intraday_efficiency'].abs() > data['intraday_efficiency'].rolling(window=10).mean()
    regime_consistency = data['volatility_regime'] == data['volatility_regime'].shift(1)
    
    # Apply cross-validation filters
    alpha = alpha * (
        momentum_pressure_alignment.astype(float) * 0.4 +
        efficiency_verification.astype(float) * 0.3 +
        regime_consistency.astype(float) * 0.3
    )
    
    return alpha
