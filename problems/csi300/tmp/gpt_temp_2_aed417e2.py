import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Novel Intraday Price-Volume Divergence with Efficiency-Weighted Momentum factor
    """
    data = df.copy()
    
    # Intraday Price-Volume Divergence components
    # High-to-Close Momentum and Low-to-Close Momentum
    data['high_to_close_momentum'] = data['high'] / data['close'] - 1
    data['low_to_close_momentum'] = data['low'] / data['close'] - 1
    data['intraday_momentum_diff'] = data['high_to_close_momentum'] - data['low_to_close_momentum']
    
    # Volume Momentum
    data['volume_5day_ma'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_momentum'] = data['volume'] / data['volume_5day_ma'] - 1
    
    # Intraday Divergence
    data['intraday_divergence'] = data['intraday_momentum_diff'] - data['volume_momentum']
    
    # Directional Divergence Strength
    data['sign_alignment'] = np.sign(data['intraday_momentum_diff']) * np.sign(data['volume_momentum'])
    data['magnitude_ratio'] = np.abs(data['intraday_momentum_diff']) / (np.abs(data['volume_momentum']) + 1e-8)
    
    # Divergence Persistence
    data['divergence_direction'] = np.sign(data['intraday_divergence'])
    divergence_persistence = []
    current_streak = 0
    for i in range(len(data)):
        if i == 0:
            divergence_persistence.append(0)
        elif data['divergence_direction'].iloc[i] == data['divergence_direction'].iloc[i-1]:
            current_streak += 1
            divergence_persistence.append(current_streak)
        else:
            current_streak = 1
            divergence_persistence.append(current_streak)
    data['divergence_persistence'] = divergence_persistence
    
    # Historical Divergence Context
    data['divergence_20day_mean'] = data['intraday_divergence'].rolling(window=20, min_periods=1).mean()
    data['divergence_deviation'] = data['intraday_divergence'] - data['divergence_20day_mean']
    data['divergence_rank'] = data['intraday_divergence'].rolling(window=20, min_periods=1).apply(
        lambda x: (x.iloc[-1] > x).mean(), raw=False
    )
    
    # Multi-Timeframe Intraday Momentum
    data['intraday_range'] = (data['high'] - data['low']) / data['close']
    data['intraday_range_3day_ma'] = data['intraday_range'].rolling(window=3, min_periods=1).mean()
    data['range_efficiency'] = data['intraday_range'] / (data['intraday_range_3day_ma'] + 1e-8)
    
    # 8-day High-to-Close Momentum consistency
    data['high_to_close_8day_std'] = data['high_to_close_momentum'].rolling(window=8, min_periods=1).std()
    data['momentum_consistency'] = 1 / (data['high_to_close_8day_std'] + 1e-8)
    
    # Momentum Acceleration
    data['momentum_acceleration'] = data['intraday_momentum_diff'].diff(3)
    
    # Efficiency-Weighted Momentum
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['prev_close']),
            np.abs(data['low'] - data['prev_close'])
        )
    )
    data['close_movement_efficiency'] = np.abs(data['close'] - data['prev_close']) / (data['true_range'] + 1e-8)
    data['intraday_efficiency'] = (data['high'] - data['low']) / (data['true_range'] + 1e-8)
    data['efficiency_weighted_signal'] = data['intraday_momentum_diff'] * data['close_movement_efficiency']
    
    # Volume-Confirmed Persistence
    data['volume_trend_strength'] = data['volume_momentum']
    data['volume_momentum_alignment'] = np.sign(data['volume_trend_strength']) * np.sign(data['intraday_momentum_diff'])
    
    # Persistence Score
    persistence_score = []
    current_persistence = 0
    for i in range(len(data)):
        if i == 0:
            persistence_score.append(0)
        elif data['volume_momentum_alignment'].iloc[i] > 0:
            current_persistence += 1
            persistence_score.append(current_persistence)
        else:
            current_persistence = 0
            persistence_score.append(current_persistence)
    data['persistence_score'] = persistence_score
    
    # Intraday Regime Detection
    # Volatility-Based Intraday Regimes
    data['intraday_volatility'] = (data['high'] - data['low']) / data['close']
    data['volatility_20day_median'] = data['intraday_volatility'].rolling(window=20, min_periods=1).median()
    data['volatility_regime'] = (data['intraday_volatility'] > data['volatility_20day_median']).astype(int)
    data['volatility_transition'] = data['volatility_regime'].diff()
    
    # Volume-Based Intraday Regimes
    data['volume_20day_ma'] = data['volume'].rolling(window=20, min_periods=1).mean()
    data['volume_clustering'] = data['volume'] / data['volume_20day_ma']
    
    # Volume regime classification
    def classify_volume_regime(clustering):
        if clustering > 1.2:
            return 2  # High
        elif clustering > 0.8:
            return 1  # Medium
        else:
            return 0  # Low
    
    data['volume_regime'] = data['volume_clustering'].apply(classify_volume_regime)
    
    # Volume regime persistence
    volume_regime_persistence = []
    current_volume_streak = 0
    for i in range(len(data)):
        if i == 0:
            volume_regime_persistence.append(0)
        elif data['volume_regime'].iloc[i] == data['volume_regime'].iloc[i-1]:
            current_volume_streak += 1
            volume_regime_persistence.append(current_volume_streak)
        else:
            current_volume_streak = 1
            volume_regime_persistence.append(current_volume_streak)
    data['volume_regime_persistence'] = volume_regime_persistence
    
    # Price-Trend Intraday Regimes
    def classify_trend_regime(momentum_diff):
        if momentum_diff > 0.02:
            return 2  # Strong Up
        elif momentum_diff > 0:
            return 1  # Weak Up
        elif momentum_diff < -0.02:
            return -2  # Strong Down
        elif momentum_diff < 0:
            return -1  # Weak Down
        else:
            return 0  # Neutral
    
    data['trend_regime'] = data['intraday_momentum_diff'].apply(classify_trend_regime)
    
    # Multi-Regime Overlap
    data['multi_regime_overlap'] = (
        data['volatility_regime'] + 
        data['volume_regime'] + 
        np.abs(data['trend_regime'])
    ) / 5.0  # Normalize
    
    # Composite Alpha Factor Integration
    # Base Signal
    data['base_signal'] = data['efficiency_weighted_signal'] * data['volume_momentum_alignment']
    
    # Divergence Multiplier and Directional Adjustment
    data['divergence_multiplier'] = 1 + np.abs(data['intraday_divergence'])
    data['directional_adjustment'] = np.sign(data['intraday_divergence']) * data['base_signal']
    
    # Regime-Adaptive Signal Processing
    data['volatility_adjustment'] = data['directional_adjustment'] * (1 + data['intraday_volatility'] / (data['volatility_20day_median'] + 1e-8))
    data['volume_regime_weight'] = data['volatility_adjustment'] * data['volume_clustering']
    
    # Trend regime filter
    def apply_trend_filter(signal, trend_regime):
        if (signal > 0 and trend_regime > 0) or (signal < 0 and trend_regime < 0):
            return signal
        else:
            return signal * 0.5  # Reduce signal strength if trend doesn't confirm
    
    data['trend_filtered_signal'] = [
        apply_trend_filter(signal, trend) 
        for signal, trend in zip(data['volume_regime_weight'], data['trend_regime'])
    ]
    
    # Final Composite Alpha
    data['persistence_enhanced'] = data['trend_filtered_signal'] * (1 + data['persistence_score'] / 10)
    
    # Regime-aligned scaling
    def regime_scaling(signal, volatility_regime, volume_regime, trend_regime):
        base_scale = 1.0
        # Higher scaling in high volatility
        if volatility_regime == 1:
            base_scale *= 1.2
        # Higher scaling in high volume
        if volume_regime == 2:
            base_scale *= 1.1
        # Higher scaling when trend confirms
        if (signal > 0 and trend_regime > 0) or (signal < 0 and trend_regime < 0):
            base_scale *= 1.15
        return signal * base_scale
    
    data['regime_aligned'] = [
        regime_scaling(sig, vol, vol_r, trend)
        for sig, vol, vol_r, trend in zip(
            data['persistence_enhanced'], 
            data['volatility_regime'],
            data['volume_regime'],
            data['trend_regime']
        )
    ]
    
    # Divergence context adjustment
    data['divergence_context_adjustment'] = data['regime_aligned'] * (0.5 + data['divergence_rank'])
    
    # Dynamic alpha with regime transition awareness
    data['regime_transition_boost'] = 1 + np.abs(data['volatility_transition']) * 0.1
    data['final_alpha'] = data['divergence_context_adjustment'] * data['regime_transition_boost']
    
    # Clean up and return
    alpha_series = data['final_alpha'].copy()
    alpha_series = alpha_series.replace([np.inf, -np.inf], np.nan)
    alpha_series = alpha_series.fillna(0)
    
    return alpha_series
