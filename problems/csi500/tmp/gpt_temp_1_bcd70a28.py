import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Timeframe Momentum Blending
    # Volatility-Normalized Momentum
    returns_3d = df['close'] / df['close'].shift(3) - 1
    returns_5d = df['close'] / df['close'].shift(5) - 1
    volatility_10d = df['close'].pct_change().rolling(window=10).std()
    normalized_3d = returns_3d / volatility_10d
    normalized_5d = returns_5d / volatility_10d
    factor_momentum = 0.6 * normalized_3d + 0.4 * normalized_5d
    
    # Momentum Acceleration Profile
    momentum_3d = df['close'] / df['close'].shift(3)
    momentum_5d = df['close'] / df['close'].shift(5)
    momentum_ratio = momentum_3d / momentum_5d
    factor_acceleration = momentum_ratio * (momentum_3d - 1)
    
    # Trend Persistence Score
    short_term_consistency = ((df['close'] - df['close'].shift(1)) * (df['close'].shift(1) - df['close'].shift(2))) > 0
    medium_term_consistency = ((df['close'] - df['close'].shift(3)) * (df['close'].shift(3) - df['close'].shift(6))) > 0
    factor_trend = short_term_consistency.astype(int) + medium_term_consistency.astype(int)
    
    # Volume-Price Regime Detection
    # Volume-Price Divergence Regime
    price_change_5d = df['close'] / df['close'].shift(5)
    volume_change_5d = df['volume'] / df['volume'].shift(5)
    divergence_indicator = (price_change_5d > 1.02) & (volume_change_5d < 0.95)
    factor_divergence = divergence_indicator.astype(int) * price_change_5d
    
    # High-Volume Momentum Efficiency
    volume_threshold = df['volume'].rolling(window=20).quantile(0.8)
    high_volume_returns = df['close'].pct_change().where(df['volume'] > volume_threshold)
    high_volume_return_avg = high_volume_returns.rolling(window=5).mean()
    current_volume_ratio = df['volume'] / df['volume'].rolling(window=20).mean()
    factor_volume_efficiency = high_volume_return_avg * current_volume_ratio
    
    # Regime-Adaptive Volume Signal
    current_range = df['high'] - df['low']
    avg_range_10d = (df['high'] - df['low']).rolling(window=10).mean()
    high_vol_regime = current_range > 1.5 * avg_range_10d
    regime_momentum = (df['close'] / df['close'].shift(5) - 1) * high_vol_regime.astype(int)
    volume_trend_3d = df['volume'] / df['volume'].shift(3)
    factor_regime_volume = regime_momentum * volume_trend_3d
    
    # Intraday Pattern Strength
    # Gap Reversal Efficiency
    overnight_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    intraday_return = (df['close'] - df['open']) / df['open']
    reversal_indicator = (overnight_gap * intraday_return) < 0
    reversal_strength = abs(overnight_gap) * abs(intraday_return)
    factor_gap_reversal = reversal_indicator.astype(int) * reversal_strength
    
    # Intraday Trend Consistency
    morning_strength = (df['high'] - df['open']) / df['open']
    afternoon_strength = (df['close'] - df['low']) / df['low']
    consistent_trend = (morning_strength * afternoon_strength) > 0
    factor_intraday_trend = consistent_trend.astype(int) * (morning_strength + afternoon_strength)
    
    # Range Efficiency Profile
    current_efficiency = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    efficiency_std_5d = current_efficiency.rolling(window=5).std()
    efficiency_mean_5d = current_efficiency.rolling(window=5).mean()
    stable_efficiency = abs(current_efficiency - efficiency_mean_5d) <= efficiency_std_5d
    factor_efficiency = stable_efficiency.astype(int) * current_efficiency
    
    # Volatility-Adaptive Signals
    # Range Breakout Intensity
    range_ratio = current_range / avg_range_10d
    breakout_signal = df['close'] > df['high'].shift(1)
    factor_breakout = breakout_signal.astype(int) * range_ratio * df['volume']
    
    # Volatility-Regime Momentum
    current_volatility = df['close'].pct_change().rolling(window=10).std()
    median_vol_10d = current_volatility.rolling(window=10).median()
    high_vol_regime_vol = current_volatility > median_vol_10d
    regime_momentum_vol = np.where(high_vol_regime_vol, 
                                  df['close'] / df['close'].shift(3) - 1,
                                  df['close'] / df['close'].shift(10) - 1)
    factor_vol_regime = regime_momentum_vol * high_vol_regime_vol.astype(int)
    
    # Support/Resistance Dynamics
    # Resistance Break Quality
    resistance_break = df['close'] / df['high'].rolling(window=20).max()
    volume_confirmation = df['volume'] > 1.2 * df['volume'].rolling(window=20).mean()
    factor_resistance = resistance_break * volume_confirmation.astype(int)
    
    # Support Bounce Quality
    support_bounce = df['close'] / df['low'].rolling(window=20).min()
    amount_confirmation = df['amount'] > 1.2 * df['amount'].rolling(window=20).mean()
    factor_support = support_bounce * amount_confirmation.astype(int)
    
    # Price-Volume Correlation Patterns
    # Volume-Price Direction Alignment
    price_trend_3d = np.sign(df['close'] - df['close'].shift(3))
    volume_trend_3d = np.sign(df['volume'] - df['volume'].shift(3))
    alignment_score = (price_trend_3d == volume_trend_3d).astype(int)
    factor_alignment = alignment_score * returns_3d
    
    # Volume-Weighted Momentum Persistence
    daily_returns = df['close'].pct_change()
    weighted_returns = daily_returns * df['volume']
    weighted_returns_3d = weighted_returns.rolling(window=3)
    persistence = weighted_returns_3d.apply(lambda x: x.corr(pd.Series(range(len(x)), index=x.index)) if len(x) == 3 else np.nan)
    factor_persistence = persistence * weighted_returns_3d.sum()
    
    # Multi-Timeframe Volume Confirmation
    short_volume_accel = df['volume'] / df['volume'].shift(3)
    medium_volume_trend = df['volume'] / df['volume'].shift(10)
    volume_confirmation_multi = (short_volume_accel > 1) & (medium_volume_trend > 1)
    factor_volume_multi = volume_confirmation_multi.astype(int) * (short_volume_accel + medium_volume_trend)
    
    # Combine all factors with equal weighting
    factors = [
        factor_momentum, factor_acceleration, factor_trend,
        factor_divergence, factor_volume_efficiency, factor_regime_volume,
        factor_gap_reversal, factor_intraday_trend, factor_efficiency,
        factor_breakout, factor_vol_regime, factor_resistance,
        factor_support, factor_alignment, factor_persistence, factor_volume_multi
    ]
    
    # Normalize each factor and combine
    combined_factor = pd.Series(0, index=df.index)
    for factor in factors:
        normalized_factor = (factor - factor.mean()) / factor.std()
        combined_factor += normalized_factor
    
    return combined_factor
