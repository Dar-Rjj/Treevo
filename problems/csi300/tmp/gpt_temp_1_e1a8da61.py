import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Asset Relative Momentum with Liquidity-Driven Regime Switching
    """
    # Calculate ATR(5) for momentum normalization
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_5 = true_range.rolling(window=5, min_periods=5).mean()
    
    # 1. Relative Momentum Components
    # Intraday momentum strength
    intraday_momentum = (df['close'] - df['open']) / atr_5
    
    # Multi-day momentum persistence (3-day)
    momentum_3d = intraday_momentum.rolling(window=3, min_periods=3).mean()
    momentum_acceleration = intraday_momentum - momentum_3d.shift(1)
    
    # 2. Liquidity Regime Detection
    # Volume profile analysis
    volume_ma_20 = df['volume'].rolling(window=20, min_periods=20).mean()
    volume_regime = df['volume'] / volume_ma_20
    
    # Volume volatility regime
    volume_volatility = df['volume'].rolling(window=10, min_periods=10).std() / volume_ma_20
    volume_regime_stable = (volume_volatility < volume_volatility.rolling(window=20).quantile(0.3)).astype(int)
    volume_regime_volatile = (volume_volatility > volume_volatility.rolling(window=20).quantile(0.7)).astype(int)
    
    # Price impact from bid-ask spread estimation
    spread_estimate = (df['high'] - df['low']) / df['close']
    normalized_spread = spread_estimate / spread_estimate.rolling(window=20, min_periods=20).mean()
    
    # 3. Cross-Asset Momentum Integration (using sector approximation via rolling cross-section)
    # Cross-sectional momentum ranking
    momentum_rank = intraday_momentum.rolling(window=5, min_periods=5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Sector-relative momentum (approximated via rolling quantiles)
    sector_momentum = intraday_momentum.rolling(window=20, min_periods=20).quantile(0.5)
    stock_sector_relative = intraday_momentum - sector_momentum
    
    # 4. Multi-Timeframe Momentum Convergence
    # Short-term (1-3 day)
    short_term_momentum = intraday_momentum.rolling(window=3, min_periods=3).mean()
    
    # Medium-term (5-10 day)
    medium_term_momentum = intraday_momentum.rolling(window=10, min_periods=10).mean()
    
    # Long-term context (15-20 day)
    long_term_momentum = intraday_momentum.rolling(window=20, min_periods=20).mean()
    
    # Momentum convergence score
    momentum_convergence = (
        (short_term_momentum > medium_term_momentum).astype(int) +
        (medium_term_momentum > long_term_momentum).astype(int) +
        (short_term_momentum > 0).astype(int)
    )
    
    # 5. Liquidity-Regime Weighted Momentum
    # High liquidity regime: amplify confirmed signals
    high_liquidity_weight = np.where(
        (volume_regime > 1.2) & (volume_regime_stable == 1),
        1.5,  # Amplify in high, stable volume
        1.0
    )
    
    # Low liquidity regime: filter noise
    low_liquidity_filter = np.where(
        (volume_regime < 0.8) | (normalized_spread > 1.3),
        0.7,  # Reduce weight in low liquidity/high spread
        1.0
    )
    
    # Volume-momentum interaction
    volume_momentum_confirmation = np.where(
        (intraday_momentum > 0) & (volume_regime > 1.1),
        1.2,  # Boost momentum confirmed by volume
        np.where(
            (intraday_momentum > 0) & (volume_regime < 0.9),
            0.8,  # Reduce momentum without volume support
            1.0
        )
    )
    
    # 6. Final Adaptive Momentum Factor
    # Base momentum component
    base_momentum = (
        0.4 * momentum_rank +
        0.3 * stock_sector_relative +
        0.3 * momentum_convergence
    )
    
    # Apply liquidity regime adjustments
    regime_adjusted_momentum = (
        base_momentum * 
        high_liquidity_weight * 
        low_liquidity_filter * 
        volume_momentum_confirmation
    )
    
    # Cross-sectional normalization
    final_factor = regime_adjusted_momentum.rolling(window=20, min_periods=20).apply(
        lambda x: (x.iloc[-1] - np.mean(x)) / np.std(x) if np.std(x) > 0 else 0,
        raw=False
    )
    
    return final_factor
