import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining price-volume divergence, gap filling propensity,
    regime-sensitive mean reversion, volume-weighted price acceleration, and 
    support/resistance breakout efficiency.
    """
    data = df.copy()
    
    # Price-Volume Divergence Momentum
    # Calculate 5-day price and volume returns
    price_return = (data['close'] / data['close'].shift(5) - 1)
    volume_return = (data['volume'] / data['volume'].shift(5) - 1)
    
    # Detect divergence patterns
    bullish_divergence = (price_return < 0) & (volume_return > 0)
    bearish_divergence = (price_return > 0) & (volume_return < 0)
    
    # Generate divergence factor with magnitude weighting
    divergence_factor = np.zeros(len(data))
    abs_price_return = np.abs(price_return)
    abs_volume_return = np.abs(volume_return)
    
    divergence_factor[bullish_divergence] = abs_price_return[bullish_divergence]
    divergence_factor[bearish_divergence] = -abs_price_return[bearish_divergence]
    
    # Strength adjustment
    min_abs_return = np.minimum(abs_price_return, abs_volume_return)
    divergence_factor *= min_abs_return
    divergence_factor = pd.Series(divergence_factor, index=data.index)
    
    # Gap Filling Propensity
    # Calculate overnight gap
    overnight_gap = (data['open'] / data['close'].shift(1) - 1)
    
    # Classify gap size
    small_gap = (np.abs(overnight_gap) < 0.01)
    medium_gap = (np.abs(overnight_gap) >= 0.01) & (np.abs(overnight_gap) <= 0.03)
    large_gap = (np.abs(overnight_gap) > 0.03)
    
    # Calculate gap filling progress
    gap_direction = np.sign(overnight_gap)
    current_close = data['close']
    gap_fill_progress = np.where(
        gap_direction > 0,  # Gap up
        (data['low'] - data['open']) / (data['close'].shift(1) - data['open']),
        (data['open'] - data['high']) / (data['open'] - data['close'].shift(1))
    )
    gap_fill_progress = np.clip(gap_fill_progress, 0, 1)
    
    # Historical filling probability (simplified)
    gap_events = overnight_gap[np.abs(overnight_gap) > 0.005]
    if len(gap_events) > 0:
        hist_fill_prob = gap_events.rolling(window=min(20, len(gap_events)), min_periods=1).apply(
            lambda x: np.mean(np.abs(x) < 0.002) if len(x) > 0 else 0.5
        )
    else:
        hist_fill_prob = pd.Series(0.5, index=data.index)
    
    # Gap factor calculation
    gap_factor = -overnight_gap * hist_fill_prob * (1 - gap_fill_progress)
    
    # Regime-Sensitive Mean Reversion
    # Calculate regime indicator (50-day trend and volatility)
    trend_strength = data['close'].rolling(window=50).apply(
        lambda x: (x[-1] - x[0]) / (np.std(x) + 1e-8) if len(x) == 50 else 0
    )
    volatility = data['close'].pct_change().rolling(window=50).std()
    
    # Classify regime
    mean_reverting_regime = (np.abs(trend_strength) < 0.5) & (volatility > volatility.quantile(0.3))
    trending_regime = np.abs(trend_strength) > 1.0
    transition_regime = ~(mean_reverting_regime | trending_regime)
    
    # Calculate rolling z-score (20-day)
    z_score = (data['close'] - data['close'].rolling(window=20).mean()) / (data['close'].rolling(window=20).std() + 1e-8)
    
    # Identify extreme levels
    overbought = z_score > 1.5
    oversold = z_score < -1.5
    
    # Regime-adjusted factor
    regime_factor = np.zeros(len(data))
    regime_weight = np.where(mean_reverting_regime, 1.0, 
                           np.where(trending_regime, 0.3, 0.6))
    
    extreme_multiplier = np.where(overbought, -np.maximum(0, z_score - 1.5),
                                np.where(oversold, np.maximum(0, -z_score - 1.5), 0))
    
    regime_factor = -z_score * regime_weight * (1 + extreme_multiplier)
    regime_factor = pd.Series(regime_factor, index=data.index)
    
    # Volume-Weighted Price Acceleration
    # Calculate price acceleration (second derivative)
    price_change = data['close'].pct_change()
    price_acceleration = price_change.diff()
    
    # Calculate volume weighting
    volume_avg = data['volume'].rolling(window=5).mean()
    volume_significance = data['volume'] / (volume_avg + 1e-8) - 1
    
    # Generate weight matrix
    volume_weight = np.where(
        (volume_significance > 0.2) & (price_acceleration * volume_significance > 0),
        1.5,
        np.where(
            (volume_significance < -0.1) & (price_acceleration * volume_significance < 0),
            0.5,
            1.0
        )
    )
    
    # Acceleration factor
    acceleration_factor = price_acceleration * volume_weight * np.abs(price_change)
    
    # Support/Resistance Breakout Efficiency
    # Define dynamic support/resistance levels
    recent_high = data['high'].rolling(window=10).max()
    recent_low = data['low'].rolling(window=10).min()
    
    # Calculate barrier touches (simplified)
    resistance_touches = (data['high'] >= recent_high * 0.995).rolling(window=10).sum()
    support_touches = (data['low'] <= recent_low * 1.005).rolling(window=10).sum()
    
    # Breakout detection
    resistance_break = (data['close'] > recent_high) & (data['volume'] > data['volume'].rolling(window=10).mean())
    support_break = (data['close'] < recent_low) & (data['volume'] > data['volume'].rolling(window=10).mean())
    
    # Breakout strength
    breakout_strength = np.where(
        resistance_break,
        (data['close'] - recent_high) / recent_high,
        np.where(
            support_break,
            (recent_low - data['close']) / recent_low,
            0
        )
    )
    
    # Volume confirmation
    volume_surge = data['volume'] / data['volume'].rolling(window=10).mean() - 1
    volume_confirmation = np.clip(volume_surge, 0, 2)
    
    # Barrier strength adjustment
    barrier_strength = np.where(
        resistance_break,
        resistance_touches / 10,
        np.where(support_break, support_touches / 10, 1)
    )
    
    # Efficiency score
    efficiency_score = breakout_strength * volume_confirmation * barrier_strength
    
    # Combine all factors with equal weighting
    combined_factor = (
        divergence_factor.fillna(0) * 0.2 +
        gap_factor.fillna(0) * 0.2 +
        regime_factor.fillna(0) * 0.2 +
        acceleration_factor.fillna(0) * 0.2 +
        efficiency_score.fillna(0) * 0.2
    )
    
    # Normalize the final factor
    if len(combined_factor) > 20:
        combined_factor = (combined_factor - combined_factor.rolling(window=20).mean()) / (combined_factor.rolling(window=20).std() + 1e-8)
    
    return combined_factor
