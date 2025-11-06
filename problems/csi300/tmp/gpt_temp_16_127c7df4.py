import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adaptive Price-Memory Factor
    Combines multi-timeframe price memory analysis with volatility regime detection
    and volume confirmation dynamics to generate adaptive trading signals.
    """
    data = df.copy()
    
    # Calculate returns and basic price metrics
    data['returns'] = data['close'].pct_change()
    data['high_low_range'] = (data['high'] - data['low']) / data['close']
    
    # 1. Multi-Timeframe Price Memory Analysis
    # Historical Influence Decay
    # 5-day return persistence (autocorrelation of 1-day returns over 5 days)
    data['return_persistence_5d'] = data['returns'].rolling(window=5).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) == 5 and not np.isnan(x).any() else 0
    )
    
    # 20-day support/resistance level persistence
    data['rolling_high_20d'] = data['high'].rolling(window=20).max()
    data['rolling_low_20d'] = data['low'].rolling(window=20).min()
    data['price_proximity_to_resistance'] = (data['rolling_high_20d'] - data['close']) / data['rolling_high_20d']
    data['price_proximity_to_support'] = (data['close'] - data['rolling_low_20d']) / data['close']
    data['support_resistance_persistence'] = (
        data['price_proximity_to_resistance'].rolling(window=5).std() + 
        data['price_proximity_to_support'].rolling(window=5).std()
    ).fillna(0)
    
    # Memory decay rate
    data['memory_decay_rate'] = (
        (1 - abs(data['return_persistence_5d'])) * 
        (1 - data['support_resistance_persistence'].rolling(window=5).mean())
    ).fillna(0.5)
    
    # Current Pattern Relevance - Fractal Analysis
    # 5-day fractal dimension
    def calc_fractal_dimension(high_low_series):
        if len(high_low_series) < 5:
            return 1.0
        range_std = np.std(high_low_series)
        range_mean = np.mean(high_low_series)
        if range_mean == 0:
            return 1.0
        return min(2.0, max(1.0, 1 + (range_std / range_mean)))
    
    data['fractal_5d'] = data['high_low_range'].rolling(window=5).apply(
        calc_fractal_dimension, raw=True
    ).fillna(1.5)
    
    # 20-day fractal dimension
    data['fractal_20d'] = data['high_low_range'].rolling(window=20).apply(
        calc_fractal_dimension, raw=True
    ).fillna(1.5)
    
    # Price path efficiency
    data['price_path_efficiency_5d'] = (
        abs(data['close'] - data['close'].shift(4)) / 
        data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()
    ).fillna(0.5)
    
    data['price_path_efficiency_20d'] = (
        abs(data['close'] - data['close'].shift(19)) / 
        data['high'].rolling(window=20).max() - data['low'].rolling(window=20).min()
    ).fillna(0.5)
    
    # Pattern recognition score
    data['pattern_score_5d'] = (
        (2 - data['fractal_5d']) * data['price_path_efficiency_5d']
    ).fillna(0)
    data['pattern_score_20d'] = (
        (2 - data['fractal_20d']) * data['price_path_efficiency_20d']
    ).fillna(0)
    
    # 2. Volatility Regime Detection
    # Average True Range calculations
    def true_range(high, low, close_prev):
        return max(high - low, abs(high - close_prev), abs(low - close_prev))
    
    data['true_range'] = [
        true_range(data['high'].iloc[i], data['low'].iloc[i], 
                  data['close'].iloc[i-1] if i > 0 else data['close'].iloc[i])
        for i in range(len(data))
    ]
    
    data['atr_5d'] = data['true_range'].rolling(window=5).mean().fillna(data['true_range'])
    data['atr_20d'] = data['true_range'].rolling(window=20).mean().fillna(data['true_range'])
    
    # Volatility ratio and regime detection
    data['volatility_ratio'] = (data['atr_5d'] / data['atr_20d']).fillna(1.0)
    
    # Regime classification
    data['volatility_regime'] = pd.cut(
        data['volatility_ratio'],
        bins=[0, 0.7, 1.3, float('inf')],
        labels=['low', 'normal', 'high']
    )
    
    # Momentum changes during regime shifts
    data['momentum_5d'] = data['close'].pct_change(5)
    data['momentum_20d'] = data['close'].pct_change(20)
    data['momentum_change'] = (
        data['momentum_5d'].rolling(window=5).std() / 
        data['momentum_20d'].rolling(window=5).std()
    ).fillna(1.0)
    
    # Volume pattern alterations
    data['volume_change_5d'] = data['volume'].pct_change(5)
    data['volume_change_20d'] = data['volume'].pct_change(20)
    data['volume_pattern_alteration'] = (
        abs(data['volume_change_5d'] - data['volume_change_20d'])
    ).fillna(0)
    
    # Transition confidence score
    data['transition_confidence'] = (
        1 / (1 + data['momentum_change'] + data['volume_pattern_alteration'])
    ).fillna(0.5)
    
    # 3. Volume Confirmation Dynamics
    # Volume acceleration
    data['volume_momentum_5d'] = (data['volume'] / data['volume'].shift(5) - 1).fillna(0)
    data['volume_momentum_20d'] = (data['volume'] / data['volume'].shift(20) - 1).fillna(0)
    data['volume_acceleration_divergence'] = (
        data['volume_momentum_5d'] - data['volume_momentum_20d']
    ).fillna(0)
    
    # Volume clustering in current price ranges
    current_price_range_low = data['close'] * 0.99
    current_price_range_high = data['close'] * 1.01
    
    def volume_clustering(volume_series, close_series, low_bound, high_bound):
        if len(volume_series) < 5:
            return 0
        in_range_mask = (close_series >= low_bound) & (close_series <= high_bound)
        if in_range_mask.sum() == 0:
            return 0
        return volume_series[in_range_mask].mean() / volume_series.mean()
    
    data['volume_clustering'] = [
        volume_clustering(
            data['volume'].iloc[max(0, i-19):i+1],
            data['close'].iloc[max(0, i-19):i+1],
            current_price_range_low.iloc[i],
            current_price_range_high.iloc[i]
        ) if i >= 19 else 0
        for i in range(len(data))
    ]
    
    # Volume efficiency during memory pattern formation
    data['volume_efficiency'] = (
        data['returns'].abs() / (data['volume'] / data['volume'].rolling(window=20).mean())
    ).fillna(0)
    
    # 4. Regime-Adaptive Signal Integration
    def regime_adaptive_signal(row):
        if row['volatility_regime'] == 'high':
            # High volatility regime - emphasize short-term components
            short_term_weight = 0.7
            medium_term_weight = 0.3
            volume_accel_dampening = 0.5
            memory_component = (
                short_term_weight * row['pattern_score_5d'] * row['return_persistence_5d'] +
                medium_term_weight * row['pattern_score_20d']
            )
            volume_component = (
                volume_accel_dampening * row['volume_acceleration_divergence'] +
                row['volume_clustering']
            )
            signal = memory_component * volume_component * row['transition_confidence']
            
        elif row['volatility_regime'] == 'low':
            # Low volatility regime - emphasize medium-term components
            short_term_weight = 0.3
            medium_term_weight = 0.7
            volume_amplification = 1.5
            memory_component = (
                short_term_weight * row['pattern_score_5d'] +
                medium_term_weight * row['pattern_score_20d'] * (1 - row['memory_decay_rate'])
            )
            volume_component = (
                volume_amplification * row['volume_acceleration_divergence'] +
                row['volume_clustering'] * row['volume_efficiency']
            )
            signal = memory_component * volume_component
            
        else:  # Normal volatility regime
            # Balanced approach
            short_term_weight = 0.5
            medium_term_weight = 0.5
            memory_component = (
                short_term_weight * row['pattern_score_5d'] * row['return_persistence_5d'] +
                medium_term_weight * row['pattern_score_20d'] * (1 - row['memory_decay_rate'])
            )
            volume_component = (
                row['volume_acceleration_divergence'] +
                row['volume_clustering']
            )
            confidence = (row['transition_confidence'] + (1 - row['memory_decay_rate'])) / 2
            signal = memory_component * volume_component * confidence
            
        return signal
    
    # Calculate final factor values
    data['factor'] = data.apply(regime_adaptive_signal, axis=1)
    
    # Normalize the factor
    data['factor_normalized'] = (
        (data['factor'] - data['factor'].rolling(window=20).mean()) / 
        data['factor'].rolling(window=20).std()
    ).fillna(0)
    
    return data['factor_normalized']
