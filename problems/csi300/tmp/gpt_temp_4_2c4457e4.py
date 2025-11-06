import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum-Adjusted Volume Breakout
    # Volume above rolling mean
    volume_mean_20 = data['volume'].rolling(window=20, min_periods=10).mean()
    volume_signal = data['volume'] / volume_mean_20
    
    # Price momentum across horizons
    momentum_5 = data['close'].pct_change(periods=5)
    momentum_10 = data['close'].pct_change(periods=10)
    
    # Weighted momentum average (5-day gets higher weight)
    weighted_momentum = 0.6 * momentum_5 + 0.4 * momentum_10
    
    # Composite factor
    volume_breakout_factor = volume_signal * weighted_momentum
    
    # Volatility-Regime Adjusted Price Range
    # Daily range ratio
    daily_range = data['high'] / data['low']
    
    # Volatility regime classification
    volatility_20 = data['close'].pct_change().rolling(window=20, min_periods=10).std()
    vol_percentile = volatility_20.rolling(window=252, min_periods=63).apply(
        lambda x: (x.iloc[-1] > np.percentile(x.dropna(), 67)) if len(x.dropna()) > 0 else np.nan
    )
    
    # Regime-specific multiplier
    regime_multiplier = np.where(vol_percentile > 0.5, 1.2, 0.8)
    range_factor = daily_range * regime_multiplier
    
    # Asymmetric Order Flow Imbalance
    # Intraday pressure signals
    close_open_gap = (data['close'] - data['open']) / data['open']
    range_efficiency = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volume timing imbalance (early/late volume ratio)
    # Using first half vs second half of day proxy (simplified)
    volume_imbalance = data['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / (x.iloc[-1] + x.iloc[0]) if len(x) >= 2 else np.nan
    )
    
    # Combined alpha
    order_flow_factor = close_open_gap * volume_imbalance * range_efficiency
    
    # Liquidity-Weighted Momentum Convergence
    # Multi-horizon momentum alignment
    momentum_1 = data['close'].pct_change(periods=1)
    momentum_3 = data['close'].pct_change(periods=3)
    momentum_5 = data['close'].pct_change(periods=5)
    
    # Count same-direction momentum signals
    same_direction = ((momentum_1 > 0) & (momentum_3 > 0) & (momentum_5 > 0)) | \
                    ((momentum_1 < 0) & (momentum_3 < 0) & (momentum_5 < 0))
    
    # Strength of momentum convergence
    momentum_strength = (momentum_1.abs() + momentum_3.abs() + momentum_5.abs()) / 3
    
    convergence_strength = same_direction.astype(float) * momentum_strength
    
    # Volume trend as liquidity proxy
    volume_trend = data['volume'].pct_change(periods=5)
    
    # Convergence-liquidity composite
    liquidity_momentum_factor = convergence_strength * volume_trend
    
    # Price-Volume Divergence Detection
    # Normalized price and volume changes
    price_volatility = data['close'].pct_change().rolling(window=20, min_periods=10).std()
    normalized_price = data['close'].pct_change(periods=5) / price_volatility.replace(0, np.nan)
    
    volume_zscore = (data['volume'] - data['volume'].rolling(window=20, min_periods=10).mean()) / \
                   data['volume'].rolling(window=20, min_periods=10).std().replace(0, np.nan)
    
    # Divergence magnitude with direction
    divergence = (normalized_price - volume_zscore) * np.sign(normalized_price)
    
    # Combine all factors with equal weights
    combined_factor = (
        volume_breakout_factor.fillna(0) +
        range_factor.fillna(0) +
        order_flow_factor.fillna(0) +
        liquidity_momentum_factor.fillna(0) +
        divergence.fillna(0)
    ) / 5
    
    return combined_factor
