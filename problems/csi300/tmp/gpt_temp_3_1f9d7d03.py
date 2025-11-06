import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Volatility-Adjusted Momentum with Volume Divergence alpha factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Volatility-Adjusted Momentum Component
    # Multi-Timeframe Raw Returns
    close = df['close']
    returns_3d = close / close.shift(3) - 1
    returns_5d = close / close.shift(5) - 1
    returns_10d = close / close.shift(10) - 1
    returns_20d = close / close.shift(20) - 1
    
    # Volatility Normalization
    daily_returns = close.pct_change()
    vol_20d = daily_returns.rolling(window=20).std()
    
    # Volatility-adjusted momentum with zero protection
    vol_adj_momentum_3d = returns_3d / (vol_20d + 1e-8)
    vol_adj_momentum_5d = returns_5d / (vol_20d + 1e-8)
    vol_adj_momentum_10d = returns_10d / (vol_20d + 1e-8)
    vol_adj_momentum_20d = returns_20d / (vol_20d + 1e-8)
    
    # Volume Divergence Component
    volume = df['volume']
    # Volume Intensity Ratios
    vol_ma_5 = volume.rolling(window=5).mean()
    vol_ma_10 = volume.rolling(window=10).mean()
    vol_ma_20 = volume.rolling(window=20).mean()
    
    vol_intensity_short = volume / (vol_ma_5 + 1e-8)
    vol_intensity_medium = volume / (vol_ma_10 + 1e-8)
    vol_intensity_long = volume / (vol_ma_20 + 1e-8)
    
    # Volume-Price Divergence
    vol_acceleration = volume / volume.shift(1) - 1
    price_momentum_5d_sign = np.sign(returns_5d)
    
    # Volume trend persistence
    vol_above_ma_20 = (volume > vol_ma_20).astype(int)
    vol_trend_persistence = vol_above_ma_20.rolling(window=5).sum()
    
    # Liquidity Normalization Component
    amount = df['amount']
    amount_ma_20 = amount.rolling(window=20).mean()
    liquidity_ratio = amount / (amount_ma_20 + 1e-8)
    liquidity_momentum = amount / amount.shift(5) - 1
    
    # Price Efficiency Signals
    high, low, open_price = df['high'], df['low'], df['open']
    daily_range_efficiency = (close - low) / ((high - low) + 1e-8)
    opening_gap_efficiency = (open_price - close.shift(1)) / (close.shift(1) + 1e-8)
    
    # Multiplicative Interaction Layer
    # Momentum-Timeframe Interactions
    momentum_alignment_ultra_long = vol_adj_momentum_3d * vol_adj_momentum_20d
    momentum_alignment_short_medium = vol_adj_momentum_5d * vol_adj_momentum_10d
    
    timeframe_consistency = (np.sign(vol_adj_momentum_3d) * 
                           np.sign(vol_adj_momentum_5d) * 
                           np.sign(vol_adj_momentum_20d))
    
    # Volume-Momentum Interactions
    vol_momentum_interaction = vol_adj_momentum_5d * vol_intensity_medium
    vol_divergence_confirmation = vol_trend_persistence * np.sign(returns_5d)
    
    # Regime Adaptation
    # Volatility regime detection
    vol_20d_rank = vol_20d.rolling(window=60).apply(lambda x: (x.iloc[-1] > x.quantile(0.7)))
    high_vol_regime = vol_20d_rank.fillna(0)
    
    # Momentum regime detection
    positive_momentum_days = (returns_20d > 0).rolling(window=10).sum()
    strong_trend_regime = (positive_momentum_days >= 7) | (positive_momentum_days <= 3)
    
    # Final Alpha Construction
    # Core multiplicative combination
    core_factor = (momentum_alignment_ultra_long * 
                   vol_intensity_medium * 
                   liquidity_ratio * 
                   (1 + vol_trend_persistence * 0.1))
    
    # Regime-based adjustments
    # High volatility: reduce weight on short-term signals
    regime_adjusted = core_factor.copy()
    regime_adjusted[high_vol_regime == 1] = (
        core_factor[high_vol_regime == 1] * 0.7 + 
        momentum_alignment_short_medium[high_vol_regime == 1] * 0.3
    )
    
    # Strong trends: emphasize timeframe alignment
    regime_adjusted[strong_trend_regime] = (
        regime_adjusted[strong_trend_regime] * 
        (1 + timeframe_consistency[strong_trend_regime] * 0.2)
    )
    
    return regime_adjusted
