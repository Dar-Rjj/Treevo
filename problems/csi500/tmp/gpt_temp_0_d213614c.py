import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a novel alpha factor combining regime-adaptive momentum, 
    multi-timeframe price-volume convergence, and direct price-volume interaction momentum.
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # 1. Regime-Adaptive Momentum Factor
    # Volatility Regime Detection
    def true_range(high, low, close_prev):
        return np.maximum(high - low, np.maximum(abs(high - close_prev), abs(low - close_prev)))
    
    # Calculate ATRs
    data['TR'] = true_range(data['high'], data['low'], data['close'].shift(1))
    data['ATR_5'] = data['TR'].rolling(window=5).mean()
    data['ATR_20'] = data['TR'].rolling(window=20).mean()
    data['vol_ratio'] = data['ATR_5'] / data['ATR_20']
    
    # Regime classification
    high_vol_regime = data['vol_ratio'] > 1.5
    low_vol_regime = data['vol_ratio'] < 0.7
    neutral_regime = ~high_vol_regime & ~low_vol_regime
    
    # Regime-specific momentum calculations
    data['momentum_2d'] = (data['close'].shift(2) / data['close'] - 1)  # Reversal for high vol
    data['momentum_5d'] = (data['close'] / data['close'].shift(5) - 1)  # Momentum for low vol
    data['momentum_3d'] = (data['close'] / data['close'].shift(3) - 1)  # Neutral regime
    
    # Volume confirmation system
    data['vol_avg_10d'] = data['volume'].rolling(window=10).mean()
    data['vol_surge'] = data['volume'] / data['vol_avg_10d']
    
    # Volume surge persistence (3-day)
    vol_up = data['vol_surge'] > 1.2
    data['vol_surge_persistence'] = vol_up.rolling(window=3).sum() / 3
    
    # Regime-adaptive volume weighting
    vol_weight = np.zeros(len(data))
    vol_weight[high_vol_regime] = 0.7
    vol_weight[low_vol_regime] = 0.3
    vol_weight[neutral_regime] = 0.5
    
    # Final regime-adaptive momentum factor
    regime_momentum = np.zeros(len(data))
    regime_momentum[high_vol_regime] = data['momentum_2d'][high_vol_regime] * (1 + data['vol_surge_persistence'][high_vol_regime])
    regime_momentum[low_vol_regime] = data['momentum_5d'][low_vol_regime] * (1 + 0.5 * data['vol_surge_persistence'][low_vol_regime])
    regime_momentum[neutral_regime] = data['momentum_3d'][neutral_regime] * (1 + data['vol_surge_persistence'][neutral_regime])
    
    regime_factor = regime_momentum * vol_weight
    
    # 2. Multi-Timeframe Price-Volume Convergence
    # Short-term convergence (1-3 days)
    data['return_3d'] = data['close'] / data['close'].shift(3) - 1
    data['vol_change_3d'] = data['volume'] / data['volume'].shift(3) - 1
    data['pv_alignment_3d'] = data['return_3d'] * data['vol_change_3d']
    
    # Short-term consistency score
    daily_return_sign = np.sign(data['close'].pct_change())
    daily_vol_sign = np.sign(data['volume'].pct_change())
    same_sign = (daily_return_sign == daily_vol_sign).astype(int)
    data['consistency_3d'] = same_sign.rolling(window=3).sum() / 3
    
    # Short-term signal
    short_term_signal = data['pv_alignment_3d'] * data['consistency_3d'] * np.abs(data['return_3d'])
    
    # Medium-term convergence (5-10 days)
    data['return_5d'] = data['close'] / data['close'].shift(5) - 1
    data['vol_trend_5d'] = data['volume'].rolling(window=5).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0)
    
    # Medium-term divergence detection
    price_up = data['return_5d'] > 0
    vol_down = data['vol_trend_5d'] < 0
    price_down = data['return_5d'] < 0
    vol_up = data['vol_trend_5d'] > 0
    
    divergence_score = np.zeros(len(data))
    divergence_score[price_up & vol_down] = -np.abs(data['return_5d'][price_up & vol_down])  # Negative divergence
    divergence_score[price_down & vol_up] = np.abs(data['return_5d'][price_down & vol_up])   # Positive divergence
    
    # Medium-term alignment (same direction)
    alignment_score = np.zeros(len(data))
    same_direction = (np.sign(data['return_5d']) == np.sign(data['vol_trend_5d']))
    alignment_score[same_direction] = data['return_5d'][same_direction] * data['vol_trend_5d'][same_direction]
    
    medium_term_signal = np.where(np.abs(divergence_score) > np.abs(alignment_score), 
                                 divergence_score, alignment_score)
    
    # Multi-timeframe integration
    short_weight = 0.4
    medium_weight = 0.6
    
    # Signal agreement amplification
    signal_agreement = np.sign(short_term_signal) == np.sign(medium_term_signal)
    agreement_multiplier = np.where(signal_agreement, 1.2, 0.8)
    
    convergence_factor = (short_weight * short_term_signal + medium_weight * medium_term_signal) * agreement_multiplier
    
    # Scale by recent price range for comparability
    price_range_5d = (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()) / data['close']
    convergence_factor = convergence_factor / (price_range_5d + 1e-8)
    
    # 3. Direct Price-Volume Interaction Momentum
    # Daily interaction product
    data['daily_return'] = data['close'].pct_change()
    data['daily_vol_change'] = data['volume'].pct_change()
    data['interaction_product'] = data['daily_return'] * data['daily_vol_change']
    
    # Cumulative momentum (3-day sum)
    data['cumulative_momentum'] = data['interaction_product'].rolling(window=3).sum()
    
    # Baseline normalization
    abs_interaction_avg = data['interaction_product'].abs().rolling(window=10).mean()
    data['normalized_momentum'] = data['cumulative_momentum'] / (abs_interaction_avg + 1e-8)
    
    # Acceleration detection (2-day change)
    data['momentum_acceleration'] = data['cumulative_momentum'] - data['cumulative_momentum'].shift(2)
    
    # Persistence multiplier
    interaction_sign = np.sign(data['interaction_product'])
    streak = np.zeros(len(data))
    for i in range(1, len(data)):
        if interaction_sign.iloc[i] == interaction_sign.iloc[i-1] and not np.isnan(interaction_sign.iloc[i]):
            streak[i] = streak[i-1] + 1
    
    persistence_multiplier = 1 + (streak * 0.1)
    
    # Enhanced signal
    enhanced_signal = data['normalized_momentum'] * data['momentum_acceleration'] * persistence_multiplier
    
    # Risk adjustment
    vol_5d = data['daily_return'].rolling(window=5).std()
    direct_factor = enhanced_signal / (vol_5d + 1e-8)
    
    # Final factor combination (equal weighting for demonstration)
    final_factor = 0.33 * regime_factor + 0.33 * convergence_factor + 0.34 * direct_factor
    
    return pd.Series(final_factor, index=data.index)
