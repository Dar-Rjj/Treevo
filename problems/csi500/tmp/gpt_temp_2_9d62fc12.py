import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Persistence-Weighted Composite Alpha Factor
    Combines factor persistence with multi-dimensional liquidity assessment
    """
    # Calculate basic price features
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    # 1. Calculate Factor Persistence
    # Compute momentum factor as base signal
    returns = close.pct_change()
    momentum_5d = close.pct_change(5)
    momentum_20d = close.pct_change(20)
    
    # Create composite momentum signal
    composite_momentum = 0.4 * momentum_5d + 0.6 * momentum_20d
    
    # Calculate signal direction streak
    signal_direction = np.sign(composite_momentum)
    streak_length = signal_direction.groupby((signal_direction != signal_direction.shift()).cumsum()).cumcount() + 1
    streak_length = streak_length * signal_direction
    
    # Compute persistence score with decay
    persistence_score = streak_length.rolling(window=10, min_periods=5).mean()
    persistence_score = persistence_score / (1 + np.abs(persistence_score))  # Normalize
    
    # 2. Assess Liquidity Robustness
    # Volume-based liquidity
    volume_ma_20 = volume.rolling(window=20, min_periods=10).mean()
    volume_liquidity = volume / volume_ma_20
    
    # Amount-based liquidity
    typical_price = (high + low + close) / 3
    effective_spread = (high - low) / typical_price
    amount_ma_20 = amount.rolling(window=20, min_periods=10).mean()
    amount_liquidity = amount / amount_ma_20
    
    # Price impact liquidity (inverse relationship)
    price_range = (high - low) / close.shift(1)
    volume_impact = volume / (amount + 1e-8)  # Avoid division by zero
    price_impact_liquidity = 1 / (1 + price_range * volume_impact)
    
    # Combine liquidity dimensions
    liquidity_composite = (
        0.4 * volume_liquidity + 
        0.3 * amount_liquidity + 
        0.3 * price_impact_liquidity
    )
    liquidity_composite = liquidity_composite.rolling(window=10, min_periods=5).mean()
    
    # 3. Generate Adaptive Alpha
    # Combine persistence and liquidity
    raw_alpha = composite_momentum * persistence_score
    liquidity_adjusted_alpha = raw_alpha * liquidity_composite
    
    # Apply regime adaptation
    # Volatility regime detection
    volatility_20d = returns.rolling(window=20, min_periods=10).std()
    volatility_60d = returns.rolling(window=60, min_periods=30).std()
    volatility_regime = volatility_20d / volatility_60d
    
    # Trend regime detection
    price_slope_20d = close.rolling(window=20, min_periods=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 5 else np.nan
    )
    volume_trend_20d = volume.rolling(window=20, min_periods=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 5 else np.nan
    )
    trend_regime = np.sign(price_slope_20d) * np.sign(volume_trend_20d)
    
    # Dynamic parameter adjustment based on regimes
    volatility_weight = 1 / (1 + np.abs(volatility_regime - 1))
    trend_weight = np.abs(trend_regime)
    
    # Final adaptive alpha with regime weights
    adaptive_alpha = liquidity_adjusted_alpha * volatility_weight * trend_weight
    
    # Apply final smoothing and normalization
    final_alpha = adaptive_alpha.rolling(window=5, min_periods=3).mean()
    final_alpha = (final_alpha - final_alpha.rolling(window=60, min_periods=30).mean()) / final_alpha.rolling(window=60, min_periods=30).std()
    
    return final_alpha
