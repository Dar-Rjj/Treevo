import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Gap Momentum with Volume Confirmation
    data['gap_size'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_direction'] = np.sign(data['gap_size'])
    data['gap_volatility'] = data['gap_size'].rolling(window=5, min_periods=3).std()
    
    # Volume analysis
    data['volume_acceleration'] = data['volume'] / data['volume'].shift(1) - 1
    data['gap_volume_alignment'] = data['gap_direction'] * np.sign(data['volume_acceleration'])
    data['volume_confirmed_gap'] = data['gap_size'] * (1 + abs(data['volume_acceleration']))
    
    # Gap persistence
    gap_direction_rolling = data['gap_direction'].rolling(window=5, min_periods=3)
    data['consecutive_gap_days'] = gap_direction_rolling.apply(
        lambda x: sum(x.diff().fillna(0) == 0) if len(x) >= 3 else 0, raw=False
    )
    
    # Gap fill probability based on volatility and trend
    data['gap_fill_probability'] = 1 - (abs(data['gap_size']) / (data['gap_volatility'] + 1e-8))
    data['gap_persistence_score'] = (data['consecutive_gap_days'] / 5) * (1 - data['gap_fill_probability'])
    data['persistence_adjusted_gap'] = data['volume_confirmed_gap'] * data['gap_persistence_score']
    
    # Intraday Trend Strength
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['plus_dm'] = np.maximum(data['high'] - data['high'].shift(1), 0)
    data['minus_dm'] = np.maximum(data['low'].shift(1) - data['low'], 0)
    data['directional_ratio'] = (data['plus_dm'] - data['minus_dm']) / (data['true_range'] + 1e-8)
    
    # Intraday efficiency
    data['intraday_range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['vwap'] = (data['amount'] / data['volume']) if 'amount' in data.columns else (data['high'] + data['low'] + data['close']) / 3
    data['vwap_deviation'] = data['vwap'] / ((data['high'] + data['low']) / 2) - 1
    data['efficiency_score'] = data['intraday_range_efficiency'] * (1 - abs(data['vwap_deviation']))
    
    # Trend persistence
    intraday_direction = np.sign(data['close'] - data['open'])
    intraday_direction_rolling = intraday_direction.rolling(window=5, min_periods=3)
    data['consecutive_direction_days'] = intraday_direction_rolling.apply(
        lambda x: sum(x.diff().fillna(0) == 0) if len(x) >= 3 else 0, raw=False
    )
    
    volume_sign_consistency = np.sign(data['volume_acceleration']).rolling(window=3, min_periods=2).apply(
        lambda x: sum(x.diff().fillna(0) == 0) if len(x) >= 2 else 0, raw=False
    )
    data['persistence_score'] = (data['consecutive_direction_days'] / 5) * (volume_sign_consistency / 3)
    
    # Multi-Timeframe Momentum
    data['price_momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['volume_ma_5d'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_momentum_5d'] = data['volume'] / data['volume_ma_5d'] - 1
    data['momentum_volume_alignment_5d'] = np.sign(data['price_momentum_5d']) * np.sign(data['volume_momentum_5d'])
    
    data['price_momentum_20d'] = data['close'] / data['close'].shift(20) - 1
    data['volume_ma_20d'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_momentum_20d'] = data['volume'] / data['volume_ma_20d'] - 1
    data['momentum_acceleration'] = data['price_momentum_5d'] - data['price_momentum_20d']
    
    # Divergence patterns
    data['bullish_divergence'] = ((data['price_momentum_5d'] < 0) & (data['volume_momentum_5d'] > 0)).astype(int)
    data['bearish_divergence'] = ((data['price_momentum_5d'] > 0) & (data['volume_momentum_5d'] < 0)).astype(int)
    data['confirmation_strength'] = (data['momentum_volume_alignment_5d'] + np.sign(data['price_momentum_20d']) * np.sign(data['volume_momentum_20d'])) / 2
    
    # Volume-Amount Order Flow
    if 'amount' in data.columns:
        data['amount_change'] = (data['amount'] - data['amount'].shift(1)) / (data['amount'].shift(1) + 1e-8)
        data['amount_volume_ratio'] = data['amount'] / (data['volume'] + 1e-8)
        data['order_flow_intensity'] = data['amount_change'] * data['volume_acceleration']
    else:
        data['order_flow_intensity'] = data['volume_acceleration']
    
    intraday_trend = np.sign(data['close'] - data['open'])
    data['price_flow_alignment'] = intraday_trend * np.sign(data['order_flow_intensity'])
    
    if 'amount' in data.columns:
        data['amount_volume_consistency'] = np.sign(data['amount_change']) * np.sign(data['volume_acceleration'])
        data['divergence_score'] = abs(data['price_flow_alignment'] - data['amount_volume_consistency'])
    else:
        data['divergence_score'] = 0
    
    # Flow persistence
    flow_direction = np.sign(data['order_flow_intensity'])
    flow_direction_rolling = flow_direction.rolling(window=5, min_periods=3)
    data['consecutive_flow_days'] = flow_direction_rolling.apply(
        lambda x: sum(x.diff().fillna(0) == 0) if len(x) >= 3 else 0, raw=False
    )
    data['flow_intensity_trend'] = data['order_flow_intensity'].rolling(window=5, min_periods=3).mean()
    data['persistence_weighted_flow'] = data['order_flow_intensity'] * (data['consecutive_flow_days'] / 5)
    
    # Regime classification
    intraday_volatility = (data['high'] - data['low']).rolling(window=5, min_periods=3).mean()
    volatility_ratio = data['gap_volatility'] / (intraday_volatility + 1e-8)
    
    data['volatility_regime'] = 1  # Normal
    data.loc[volatility_ratio > 1.5, 'volatility_regime'] = 2  # High
    data.loc[volatility_ratio < 0.7, 'volatility_regime'] = 0  # Low
    
    # Regime adjustments
    regime_adjustments = np.ones(len(data))
    regime_adjustments[data['volatility_regime'] == 2] = 1 / (volatility_ratio[data['volatility_regime'] == 2] + 1e-8)  # Dampen high volatility
    regime_adjustments[data['volatility_regime'] == 0] = 1.2  # Enhance low volatility
    
    # Time decay weighting
    decay_weights = np.exp(-np.arange(len(data)) * np.log(2) / 8)  # 8-day half-life
    decay_weights = decay_weights[::-1]  # Reverse for recent emphasis
    
    # Base momentum components
    gap_momentum = data['persistence_adjusted_gap']
    intraday_trend = data['efficiency_score'] * data['persistence_score']
    multi_timeframe = data['momentum_acceleration'] * data['confirmation_strength']
    
    # Divergence integration
    divergence_signal = (data['bullish_divergence'] - data['bearish_divergence']) * data['divergence_score']
    flow_alignment = data['order_flow_intensity'] * data['price_flow_alignment']
    
    # Cross-dimensional consistency
    direction_agreement = (
        np.sign(gap_momentum) + np.sign(intraday_trend) + np.sign(multi_timeframe)
    ) / 3
    
    # Composite factor assembly
    base_composite = (
        0.3 * gap_momentum +
        0.25 * intraday_trend +
        0.25 * multi_timeframe +
        0.1 * divergence_signal +
        0.1 * flow_alignment
    )
    
    # Apply regime adjustments and consistency weighting
    final_factor = base_composite * regime_adjustments * (1 + 0.5 * abs(direction_agreement))
    
    # Apply time decay weighting to recent values
    decayed_factor = final_factor * decay_weights[:len(final_factor)]
    
    # Normalize and return
    factor_series = pd.Series(decayed_factor, index=data.index)
    return factor_series
